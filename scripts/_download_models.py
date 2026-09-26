#!/usr/bin/env python
"""Share model downloader for rag-kit (used by download-models.bat)."""
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

model_dir = os.environ.get("MODEL_DIR", os.path.expanduser("~/models"))
mirror = os.environ.get("HF_ENDPOINT", "")
if mirror:
    os.environ["HF_ENDPOINT"] = mirror
    print(f"Using mirror: {mirror}")

success = []
failed = []

RELEASE_TAG = "v0.1.0"
RELEASE_BASE = f"https://github.com/lychee888/Rag-Kit/releases/download/{RELEASE_TAG}"

MIN_SIZES = {
    "embedding-model.tar.gz": 400_000_000,
    "easyocr-models.tar.gz": 80_000_000,
    "smolvlm-model.tar.gz": 250_000_000,
}


def fetch_verify(url, dest, min_bytes, label):
    """Download with size check + gzip integrity check."""
    tmp = dest.with_suffix(".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [*] Fetching {label} ...")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:
        print(f"  FAILED: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    if tmp.stat().st_size < min_bytes:
        print(f"  FAILED: {label} too small ({tmp.stat().st_size} bytes)")
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    try:
        with tarfile.open(tmp, "r:gz") as tf:
            tf.getmembers()
    except Exception as e:
        print(f"  FAILED: {label} archive corrupt: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    with tarfile.open(tmp, "r:gz") as tf:
        tf.extractall(dest)
    try:
        tmp.unlink()
    except OSError:
        pass
    print("  OK")
    return True


try:
    import easyocr
    easyocr_ok = True
except Exception:
    easyocr_ok = False

# 1. Embedding model
print("\n[1/3] Embedding model: paraphrase-multilingual-MiniLM-L12-v2")
embed_dir = (
    Path(model_dir)
    / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    / "snapshots"
    / "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
)
if (embed_dir / "model.safetensors").exists():
    print("  already present")
    success.append("embedding")
else:
    ok = fetch_verify(
        f"{RELEASE_BASE}/embedding-model.tar.gz",
        embed_dir,
        MIN_SIZES["embedding-model.tar.gz"],
        "embedding model",
    )
    if not ok:
        try:
            from huggingface_hub import snapshot_download
            print("  Falling back to HuggingFace hub...")
            snapshot_download(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                cache_dir=model_dir,
                resume_download=True,
            )
            print("  OK (hub)")
            success.append("embedding")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append("embedding")
    else:
        success.append("embedding")

# 2. EasyOCR
print("\n[2/3] EasyOCR models: ch_sim + en")
easy_dir = Path.home() / ".EasyOCR" / "model"
if (easy_dir / "zh_sim_g2.pth").exists():
    print("  already present")
    success.append("ocr")
else:
    ok = fetch_verify(
        f"{RELEASE_BASE}/easyocr-models.tar.gz",
        easy_dir,
        MIN_SIZES["easyocr-models.tar.gz"],
        "easyocr models",
    )
    if not ok and easyocr_ok:
        try:
            import easyocr
            print("  Falling back to EasyOCR CDN...")
            easyocr.Reader(
                ["ch_sim", "en"],
                gpu=False,
                download_enabled=True,
                model_storage_directory=str(easy_dir),
            )
            print("  OK (cdn)")
            success.append("ocr")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append("ocr")
    elif ok:
        success.append("ocr")
    else:
        failed.append("ocr")

# 3. SmolVLM (optional)
skip_vlm = os.environ.get("SKIP_VLM", "0")
if skip_vlm != "1":
    print("\n[3/3] VLM model: SmolVLM-256M-Instruct")
    vlm_dir = Path(model_dir) / "models--HuggingFaceTB--SmolVLM-256M-Instruct" / "snapshots" / "manual"
    if (vlm_dir / "model.safetensors").exists():
        print("  already present")
        success.append("vlm")
    else:
        ok = fetch_verify(
            f"{RELEASE_BASE}/smolvlm-model.tar.gz",
            vlm_dir,
            MIN_SIZES["smolvlm-model.tar.gz"],
            "SmolVLM",
        )
        if not ok:
            try:
                from huggingface_hub import snapshot_download
                print("  Falling back to HuggingFace hub...")
                snapshot_download(
                    "HuggingFaceTB/SmolVLM-256M-Instruct",
                    cache_dir=model_dir,
                    resume_download=True,
                )
                print("  OK (hub)")
                success.append("vlm")
            except Exception as e:
                print(f"  FAILED: {e}")
                print("  (VLM is optional — rag-kit works without it)")
                failed.append("vlm")
        else:
            success.append("vlm")
else:
    print("\n[3/3] Skipping VLM (user chose not to download)")

print(f"\n{'=' * 60}")
print(f"  Download summary:")
print(f"    Succeeded: {', '.join(success) if success else 'none'}")
print(f"    Failed:    {', '.join(failed) if failed else 'none'}")
print(f"  Models cached in: {model_dir}")
print(f"{'=' * 60}")
if failed and not success:
    sys.exit(1)
elif failed:
    print("WARNING: Some models failed (non-critical if rag-kit can still work)")

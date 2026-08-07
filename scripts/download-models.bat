@echo off
setlocal enabledelayedexpansion
title rag-kit Model Downloader (Windows)

REM ============================================================
REM   rag-kit - Model Downloader (Windows)
REM   Pre-downloads pinned models for fully offline rag-kit use.
REM   Prefers pinned GitHub Release tarballs (with integrity
REM   checks); falls back to the HuggingFace hub if unavailable.
REM ============================================================
echo.
echo ============================================================
echo   rag-kit - Model Downloader
echo   Prefers pinned GitHub Release tarballs (fast, version-locked)
echo ============================================================
echo.
set "MODEL_DIR=%USERPROFILE%\models"

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.10+ first.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Optionally set China mirror (for HuggingFace fallback below)
set "HF_MIRROR="
if not defined RAG_KIT_BATCH (
    set /p HF_CHOICE="Use China mirror (hf-mirror.com) for HF fallback? [y/N]: "
    if /i "!HF_CHOICE!"=="y" (
        set "HF_ENDPOINT=https://hf-mirror.com"
        echo Using mirror: !HF_ENDPOINT!
        echo.
    )
)

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

echo.
echo Models to download:
echo   1. paraphrase-multilingual-MiniLM-L12-v2 (embedding, ~470 MB)
echo   2. EasyOCR ch_sim + en (OCR, ~100 MB download)
echo   3. SmolVLM-256M-Instruct (VLM, ~500 MB) - optional
echo.
REM Only prompt if the caller didn't already specify (e.g. via installer).
if not defined SKIP_VLM (
    set /p DOWNLOAD_VLM="Download SmolVLM? [Y/n]: "
    if /i "!DOWNLOAD_VLM!"=="n" (
        set "SKIP_VLM=1"
    ) else (
        set "SKIP_VLM=0"
    )
) else (
    echo   VLM choice inherited from environment (SKIP_VLM=!SKIP_VLM!)
)

echo.
echo Downloading models to: %MODEL_DIR%
echo This may take a while on first run...
echo.

REM ---- Python download script ----
set "SCRIPT=%TEMP%\rag_kit_download.py"
(
echo import os, sys, tarfile, urllib.request
echo from pathlib import Path
echo.
echo model_dir = os.environ.get("MODEL_DIR", os.path.expanduser("~/models"))
echo mirror = os.environ.get("HF_ENDPOINT", "")
echo if mirror:
echo     os.environ["HF_ENDPOINT"] = mirror
echo     print(f"Using mirror: {mirror}")
echo.
echo success = []
echo failed = []
echo.
echo RELEASE_TAG = "v0.1.0"
echo RELEASE_BASE = f"https://github.com/jarvis959/Rag-Kit/releases/download/{RELEASE_TAG}"
echo.
echo MIN_SIZES = {
echo     "embedding-model.tar.gz": 400_000_000,
echo     "easyocr-models.tar.gz": 80_000_000,
echo     "smolvlm-model.tar.gz": 250_000_000,
echo }
echo.
echo def fetch_verify(url, dest, min_bytes, label):
echo     """Download with size check + gzip integrity check."""
echo     tmp = dest.with_suffix(".part")
echo     tmp.parent.mkdir(parents=True, exist_ok=True)
echo     print(f"  [*] Fetching {label} ...")
echo     try:
echo         urllib.request.urlretrieve(url, tmp)
echo     except Exception as e:
echo         print(f"  FAILED: {e}")
echo         try: tmp.unlink()
echo         except OSError: pass
echo         return False
echo     if tmp.stat().st_size < min_bytes:
echo         print(f"  FAILED: {label} too small ({tmp.stat().st_size} bytes)")
echo         try: tmp.unlink()
echo         except OSError: pass
echo         return False
echo     try:
echo         with tarfile.open(tmp, "r:gz") as tf:
echo             tf.getmembers()
echo     except Exception as e:
echo         print(f"  FAILED: {label} archive corrupt: {e}")
echo         try: tmp.unlink()
echo         except OSError: pass
echo         return False
echo     with tarfile.open(tmp, "r:gz") as tf:
echo         tf.extractall(dest)
echo     try: tmp.unlink()
echo     except OSError: pass
echo     print("  OK")
echo     return True
echo.
echo try:
echo     import easyocr
echo     easyocr_ok = True
echo except Exception:
echo     easyocr_ok = False
echo.
echo # 1. Embedding model
echo print("\n[1/3] Embedding model: paraphrase-multilingual-MiniLM-L12-v2")
echo embed_dir = Path(model_dir) / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2" / "snapshots" / "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
echo if (embed_dir / "model.safetensors").exists():
echo     print("  already present")
echo     success.append("embedding")
echo else:
echo     ok = fetch_verify(f"{RELEASE_BASE}/embedding-model.tar.gz", embed_dir, MIN_SIZES["embedding-model.tar.gz"], "embedding model")
echo     if not ok:
echo         try:
echo             from huggingface_hub import snapshot_download
echo             print("  Falling back to HuggingFace hub...")
echo             snapshot_download(
echo                 "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
echo                 cache_dir=model_dir, resume_download=True,
echo             )
echo             print("  OK (hub)")
echo             success.append("embedding")
echo         except Exception as e:
echo             print(f"  FAILED: {e}")
echo             failed.append("embedding")
echo     else:
echo         success.append("embedding")
echo.
echo # 2. EasyOCR
echo print("\n[2/3] EasyOCR models: ch_sim + en")
echo easy_dir = Path.home() / ".EasyOCR" / "model"
echo if (easy_dir / "zh_sim_g2.pth").exists():
echo     print("  already present")
echo     success.append("ocr")
echo else:
echo     ok = fetch_verify(f"{RELEASE_BASE}/easyocr-models.tar.gz", easy_dir, MIN_SIZES["easyocr-models.tar.gz"], "easyocr models")
echo     if not ok and easyocr_ok:
echo         try:
echo             import easyocr
echo             print("  Falling back to EasyOCR CDN...")
echo             easyocr.Reader(["ch_sim", "en"], gpu=False, download_enabled=True,
echo                             model_storage_directory=str(easy_dir))
echo             print("  OK (cdn)")
echo             success.append("ocr")
echo         except Exception as e:
echo             print(f"  FAILED: {e}")
echo             failed.append("ocr")
echo     elif ok:
echo         success.append("ocr")
echo     else:
echo         failed.append("ocr")
echo.
echo # 3. SmolVLM (optional)
echo skip_vlm = os.environ.get("SKIP_VLM", "0")
echo if skip_vlm != "1":
echo     print("\n[3/3] VLM model: SmolVLM-256M-Instruct")
echo     vlm_dir = Path(model_dir) / "models--HuggingFaceTB--SmolVLM-256M-Instruct" / "snapshots" / "manual"
echo     if (vlm_dir / "model.safetensors").exists():
echo         print("  already present")
echo         success.append("vlm")
echo     else:
echo         ok = fetch_verify(f"{RELEASE_BASE}/smolvlm-model.tar.gz", vlm_dir, MIN_SIZES["smolvlm-model.tar.gz"], "SmolVLM")
echo         if not ok:
echo             try:
echo                 from huggingface_hub import snapshot_download
echo                 print("  Falling back to HuggingFace hub...")
echo                 snapshot_download(
echo                     "HuggingFaceTB/SmolVLM-256M-Instruct",
echo                     cache_dir=model_dir, resume_download=True,
echo                 )
echo                 print("  OK (hub)")
echo                 success.append("vlm")
echo             except Exception as e:
echo                 print(f"  FAILED: {e}")
echo                 print("  (VLM is optional — rag-kit works without it)")
echo                 failed.append("vlm")
echo         else:
echo             success.append("vlm")
echo else:
echo     print("\n[3/3] Skipping VLM (user chose not to download)")
echo.
echo print(f"\n{'='*60}")
echo print(f"  Download summary:")
echo print(f"    Succeeded: {', '.join(success) if success else 'none'}")
echo print(f"    Failed:    {', '.join(failed) if failed else 'none'}")
echo print(f"  Models cached in: {model_dir}")
echo print(f"{'='*60}")
echo if failed and not success:
echo     sys.exit(1)
echo elif failed:
echo     print("WARNING: Some models failed (non-critical if rag-kit can still work)")
) > "%SCRIPT%"

setlocal
if defined HF_ENDPOINT set "HF_ENDPOINT=%HF_ENDPOINT%"
set "MODEL_DIR=%MODEL_DIR%"
set "SKIP_VLM=%SKIP_VLM%"

python "%SCRIPT%"
set "DL_EXIT=%ERRORLEVEL%"
del "%SCRIPT%" 2>nul

echo.
if %DL_EXIT% NEQ 0 (
    echo ERROR: Model download failed. Check the errors above.
    echo You may need to try again or use a VPN/proxy.
    pause
    exit /b 1
)

echo ============================================================
echo   Download Complete
echo ============================================================
echo Models cached in: %MODEL_DIR%
echo You can now install rag-kit offline:
echo   scripts\install-windows.bat
echo.
REM Pause only when run interactively (not when called from the installer).
if defined RAG_KIT_BATCH (
    exit /b 0
)
pause
exit /b 0

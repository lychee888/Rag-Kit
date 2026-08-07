"""Verify which formats actually work in the CURRENT rag_kit.

Creates real sample files, runs ingest_file() on each, and where possible
pushes through the embedding + vector store to prove end-to-end.
"""
import sys, tempfile, time, traceback
from pathlib import Path

def log(m):
    print(m)

TMP = Path(tempfile.mkdtemp(prefix="ragkit_fmt_test_"))

# --- create sample files ---

# 1) xlsx (needs openpyxl)
try:
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["Part", "Qty", "Note"])
    ws.append(["C-MORE Gooseneck Mount", 3, "for Ruger 10/22"])
    ws.append(["Red Dot Sight", 2, "astigmatism friendly"])
    xlsx = TMP / "sample_inventory.xlsx"
    wb.save(str(xlsx))
    log("created xlsx")
except Exception as e:
    log(f"xlsx create FAIL: {e}")
    xlsx = None

# 2) pptx (needs python-pptx)
try:
    from pptx import Presentation
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Kern Optic Launch Plan"
    body = slide.placeholders[1]
    body.text = "Q3 release of the new gooseneck mount. Price point $89."
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    tb = s2.shapes.add_textbox(0, 0, 5000000, 3000000)
    tb.text = "Manufacturing partner: Shenzhen CNC. Lead time 4 weeks."
    pptx = TMP / "sample_deck.pptx"
    prs.save(str(pptx))
    log("created pptx")
except Exception as e:
    log(f"pptx create FAIL: {e}")
    pptx = None

# 3) image with text (needs PIL + easyocr)
try:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (900, 300), "white")
    d = ImageDraw.Draw(img)
    for i, txt in enumerate(["KERN OPTICS", "Invoice #1042", "Total: $1,357.48"]):
        d.text((30, 40 + i * 90), txt, fill="black")
    png = TMP / "sample_invoice.png"
    img.save(str(png))
    log("created png (white bg, black text)")
except Exception as e:
    log(f"png create FAIL: {e}")
    png = None

md = TMP / "sample_note.md"
md.write_text("# Gooseneck Mount Specs\n\nThe C-MORE gooseneck mount weighs 3.2 oz and fits Picatinny rails.\n", encoding="utf-8")
txt = TMP / "sample_note.txt"
txt.write_text("This is a plain text test document for the RAG pipeline.\n", encoding="utf-8")

import rag_kit
from rag_kit.ingest import ingest_file, SUPPORTED_EXTENSIONS
log(f"\nSUPPORTED_EXTENSIONS (n={len(SUPPORTED_EXTENSIONS)}): {SUPPORTED_EXTENSIONS}")
log(f"rag_kit {getattr(rag_kit,'__version__','?')}")

results = {}
for f in (md, txt, xlsx, pptx, png):
    if f is None or not f.exists():
        continue
    try:
        t0 = time.time()
        chunks = ingest_file(f, chunk_size=512, chunk_overlap=64)
        dt = time.time() - t0
        n_chunks = len(chunks)
        total_chars = sum(len(c["text"]) for c in chunks)
        sample = chunks[0]["text"][:90] if chunks else ""
        results[f.suffix] = {"file": f.name, "chunks": n_chunks, "chars": total_chars, "secs": round(dt, 2), "sample": sample}
        log(f"\n** {f.suffix:6s} {f.name}: {n_chunks} chunks, {total_chars} chars in {dt:.2f}s")
        if chunks:
            log(f"    first chunk: {sample!r}")
        else:
            log("    !! NO CHUNKS (empty extraction)")
    except Exception as e:
        results[f.suffix] = {"file": f.name, "error": f"{type(e).__name__}: {e}"}
        log(f"\n** {f.suffix:6s} {f.name}: EXC {type(e).__name__}: {e}")
        tb = traceback.format_exc().splitlines()
        log("    " + "\n    ".join(tb[-3:]))

# --- embedding + store probe (needs model, may download on first run) ---
print("\n=== embedding probe (may download model on first run) ===")
try:
    from rag_kit.embed import EmbeddingEngine
    eng = EmbeddingEngine()
    log(f"embedding model: {eng.model_name}, dim={eng.dimension}")
    test_texts = ["red dot sight for rifle", "gooseneck mount brass", "unrelated astronomy note"]
    vecs = eng.embed_texts(test_texts)
    import numpy as np
    log(f"embedded {len(vecs)} texts -> shape {np.array(vecs).shape}, dtype {np.array(vecs).dtype}")
    log(f"norm check (should be ~1.0): {float(np.linalg.norm(vecs[0])):.4f}")
    import lancedb
    db = lancedb.connect(str(TMP / "lancedb_probe"))
    tb = db.create_table("documents", data=[{"id": "a", "text": "red dot sight", "vector": list(vecs[0])}])
    hit = tb.search(list(vecs[2])).limit(3).to_list()
    log(f"lancedb probe OK: rows={tb.count_rows()}, search returned {len(hit)} row(s)")
    log("EMBED+STORE: WORKS")
except Exception as e:
    log(f"EMBED+STORE probe FAIL: {type(e).__name__}: {e}")
    log(traceback.format_exc().splitlines()[-3])

print("\n=== SUMMARY ===")
for ext, r in results.items():
    print(f"  {ext:6s}: " + ", ".join(f"{k}={v}" for k, v in r.items()))

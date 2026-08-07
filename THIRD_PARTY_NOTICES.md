# Third-Party Notices

rag-kit bundles the following pre-trained model weights and **re-distributes them as
GitHub release assets** (`scripts/download-models.*` fetch these tarballs). All three
are licensed under the **Apache License 2.0**, which permits redistribution and
re-hosting provided (a) recipients receive a copy of the license and (b) original
copyright and attribution notices are retained. The full license text is in
[`licenses/APACHE-2.0.txt`](licenses/APACHE-2.0.txt).

The weights are bundled unmodified.

---

## Embedding model — `paraphrase-multilingual-MiniLM-L12-v2`

- **Author / copyright**: Nils Reimers & the sentence-transformers contributors (UKP Lab, TU Darmstadt), based on Microsoft **MiniLM**
- **Source**: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **License**: Apache-2.0
- **Asset**: `embedding-model.tar.gz` → unpacked to `~/models/`

## Vision-language model — `SmolVLM-256M-Instruct`

- **Author / copyright**: Hugging Face Inc. (Hugging Face TB), based on **SmolLM2**
- **Source**: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
- **License**: Apache-2.0
- **Asset**: `smolvlm-model.tar.gz` → unpacked to `~/models/`

## OCR models — EasyOCR weights (`craft_mlt_25k.pth`, `zh_sim_g2.pth`)

- **Author / copyright**: Jaided AI (EasyOCR)
- **Source**: https://www.jaided.ai/easyocr/modelhub/ · https://github.com/JaidedAI/EasyOCR
- **Note**: `craft_mlt_25k.pth` is the CRAFT text-detection network re-trained by Jaided; original CRAFT is by Clova AI Research (NAVER Clova)
- **License**: Apache-2.0
- **Assets**: `easyocr-models.tar.gz` → unpacked to `~/.EasyOCR/model/`

---

Apache License, Version 2.0 — key obligations for our re-distribution:
- Give recipients a copy of the license (see `licenses/APACHE-2.0.txt`).
- Retain the copyright / attribution notices above.
- No changes were made to any model weights.

# Reference Papers — Financial Document Forgery Detection

All papers are open-access from arXiv. Organized by relevance to our pipeline.

---

## Visual Stream / CNN Forensics

| # | File | Paper | Why Relevant |
|---|---|---|---|
| 06 | `06_EdgeDoc_CNN_Transformer_IDForgery_2210.00586.pdf` | **EdgeDoc** (George & Marcel, 2022) — arXiv:2210.00586 | Hybrid CNN + Transformer for ID document forgery detection. Closest match to our visual forensic CNN stream |
| 08 | `08_ImageSplicing_MultiStreamFusion_2209.00788.pdf` | **Multi-Stream Fusion Network for Splicing Detection** (2022) — arXiv:2209.00788 | Multi-stream architecture (parallel processing of visual channels) — mirrors our ELA + Noise + JPEG Ghost multi-detector approach |
| 09 | `09_CopyMoveForgery_IMNet_2404.01717.pdf` | **IMNet — Copy-Move Forgery via Inconsistency Mining** (2024) — arXiv:2404.01717 | Object-level copy-move detection using prototype inconsistency — relevant to our RANSAC copy-paste detector |
| 05 | `05_ELA_Forensics_arXiv_2107.09669.pdf` | **ELA-based Forensics** (2021) — arXiv:2107.09669 | Error Level Analysis technique — the direct algorithm behind our `ela_detector.py` |

---

## OCR Ensemble / Text Stream

| # | File | Paper | Why Relevant |
|---|---|---|---|
| 01 | `01_LayoutLM_arXiv_1912.13318.pdf` | **LayoutLM** (Xu et al., 2020) — arXiv:1912.13318 | Multi-modal document understanding combining text + layout + image; future replacement for our fixed OCR ensemble |
| 02 | `02_LayoutLMv2_arXiv_2012.14740.pdf` | **LayoutLMv2** (Xu et al., 2021) — arXiv:2012.14740 | Improved spatial-aware pre-training for document AI; recommended citation in Related Work |
| 03 | `03_LayoutLMv3_arXiv_2204.08387.pdf` | **LayoutLMv3** (Huang et al., 2022) — arXiv:2204.08387 | Unified text + image pre-training; state-of-the-art document understanding model |
| 11 | `11_ReceiptUnderstanding_Benchmark_2103.10213.pdf` | **CORD — Receipt Understanding Benchmark** (Park et al.) — arXiv:2103.10213 | Public receipt dataset with entity annotations; can benchmark our OCR entity extraction |

---

## Multi-Modal Fusion / Full Pipeline

| # | File | Paper | Why Relevant |
|---|---|---|---|
| 04 | `04_DONUT_DocUnderstanding_arXiv_2111.15664.pdf` | **DONUT** (Kim et al., 2022) — arXiv:2111.15664 | OCR-free document understanding via end-to-end transformer; shows alternative to OCR ensemble stage |
| 07 | `07_ForgeryGPT_ExplainableDetection.pdf` | **ForgeryGPT** (2024) — arXiv:2406.01234 | LLM-based explainable forgery detection; relevant to our evidence/interpretation output |

---

## Survey / Benchmark

| # | File | Paper | Why Relevant |
|---|---|---|---|
| 10 | `10_DocumentForensics_Survey_2304.06279.pdf` | **Survey: Document Forgery Detection** (2023) — arXiv:2304.06279 | Comprehensive literature review; cite in Related Work section |
| 12 | `12_NIST_ImageForensics_MFC_arXiv_1912.03749.pdf` | **NIST MFC Dataset** (2019) — arXiv:1912.03749 | Standard forensics benchmark dataset; useful for evaluation comparison |

---

## How they map to our Fusion Equation

```
S = 0.68 × S_visual + 0.32 × S_text
```

- **S_visual** → papers 05, 06, 08, 09 (ELA, EdgeDoc, multi-stream CNN, copy-move)
- **S_text** → papers 01, 02, 03, 11 (LayoutLM family, CORD benchmark)
- **Fusion** → papers 04, 07 (DONUT, ForgeryGPT show alternative fusion strategies)
- **Literature context** → papers 10, 12 (survey + benchmark)

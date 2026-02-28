"""
ocr/ocr_engine.py
Triple OCR Consensus Engine
Tesseract + EasyOCR + PaddleOCR — parallel execution, majority-vote entities
"""

import re
import cv2
import numpy as np
import concurrent.futures
from pathlib import Path

# ── Entity extraction helpers ────────────────────────────────────────────────

# Indian GST: 15-char alphanumeric (2 digit state + 10 PAN + 1 entity + Z + 1 check)
GST_PATTERN_INDIAN  = re.compile(
    r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b'
)
# Generic Tax-ID: US/CORD-style  XX-XX-XXXX
GST_PATTERN_GENERIC = re.compile(r'\b\d{3}-\d{2}-\d{4}\b|\b\d{2}-\d{2}-\d{4}\b')
# Invoice/Bill number
INVOICE_PATTERN     = re.compile(r'(?:Invoice|Bill|Receipt|No\.?|#)\s*[:\-]?\s*([A-Z0-9\-/]+)', re.IGNORECASE)
# Single decimal numbers
DECIMAL_PATTERN     = re.compile(r'\b\d{1,8}[.,]\d{2}\b')


def _normalise(text: str) -> str:
    """Remove spaces and normalise commas→dots for number parsing."""
    text = text.replace(',', '.')
    return text


def _extract_entities_from_text(raw_text: str) -> dict:
    """
    Parse OCR text to extract financial entities and identifiers.
    Handles both Indian GST invoices and CORD/SROIE-style receipts.
    """
    norm = _normalise(raw_text)

    # ── GST / Tax ID ──
    gst = ""
    m = GST_PATTERN_INDIAN.search(norm.upper())
    if m:
        gst = m.group(0)
    else:
        m = GST_PATTERN_GENERIC.search(norm)
        if m:
            gst = m.group(0)

    # ── Invoice number ──
    invoice = ""
    m = INVOICE_PATTERN.search(raw_text)
    if m:
        invoice = m.group(1).strip()

    # ── Financial values from SUMMARY / TOTAL block ──
    net_subtotal = vat_percent = vat_amount = gross_total = None

    # VAT / Tax percent
    vp = re.search(r'(\d{1,2})\s*%', norm)
    if vp:
        vat_percent = float(vp.group(1))

    # Try to find the SUMMARY block
    summary_start = norm.upper().find("SUMMARY")
    if summary_start == -1:
        summary_start = norm.upper().find("TOTAL")

    if summary_start != -1:
        summary = norm[summary_start:]
        # Find all decimals in summary lines
        lines = summary.split('\n')
        for line in lines:
            if re.search(r'total', line, re.IGNORECASE):
                nums = DECIMAL_PATTERN.findall(line)
                nums = [float(n.replace(',', '.')) for n in nums]
                if len(nums) >= 3:
                    net_subtotal = nums[0]
                    vat_amount   = nums[1]
                    gross_total  = nums[2]
                    break
                elif len(nums) == 1:
                    gross_total  = nums[0]

    return {
        "net_subtotal": net_subtotal,
        "vat_percent":  vat_percent,
        "vat_amount":   vat_amount,
        "gross_total":  gross_total,
        "gst_number":   gst,
        "invoice_number": invoice,
    }


# ── OCR Engines ─────────────────────────────────────────────────────────────

def _run_tesseract(image_path: str) -> dict:
    try:
        import pytesseract
        from PIL import Image as PILImage
        img   = PILImage.open(image_path)
        text  = pytesseract.image_to_string(img)
        entities = _extract_entities_from_text(text)
        entities["engine"] = "tesseract"
        entities["raw_text"] = text
        return entities
    except Exception as e:
        return {"engine": "tesseract", "error": str(e), "raw_text": ""}


# ── Global Singletons for OCR Models (Lazy Loading) ─────────────────────────
_EASYOCR_READER = None
_PADDLE_OCR = None

def _get_easyocr():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _EASYOCR_READER

def _get_paddleocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        import sys
        # PaddleOCR tries to parse sys.argv on init, which can crash in Flask/Subagent
        _old_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            from paddleocr import PaddleOCR
            # removed show_log=False as it causes __init__ error in this version
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang='en')
        finally:
            sys.argv = _old_argv
    return _PADDLE_OCR


def _run_tesseract(image_path: str) -> dict:
    try:
        import pytesseract
        from PIL import Image as PILImage
        img   = PILImage.open(image_path)
        text  = pytesseract.image_to_string(img)
        entities = _extract_entities_from_text(text)
        entities["engine"] = "tesseract"
        entities["raw_text"] = text
        return entities
    except Exception as e:
        return {"engine": "tesseract", "error": str(e), "raw_text": ""}


def _run_easyocr(image_path: str) -> dict:
    try:
        reader = _get_easyocr()
        results = reader.readtext(image_path, detail=0, paragraph=True)
        text = "\n".join(results)
        entities = _extract_entities_from_text(text)
        entities["engine"] = "easyocr"
        entities["raw_text"] = text
        return entities
    except Exception as e:
        return {"engine": "easyocr", "error": str(e), "raw_text": ""}


def _run_paddleocr(image_path: str) -> dict:
    try:
        ocr = _get_paddleocr()
        result = ocr.ocr(image_path, cls=True)
        lines = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    lines.append(line[1][0])
        text = "\n".join(lines)
        entities = _extract_entities_from_text(text)
        entities["engine"] = "paddleocr"
        entities["raw_text"] = text
        return entities
    except Exception as e:
        return {"engine": "paddleocr", "error": str(e), "raw_text": ""}


# ── Majority Vote ────────────────────────────────────────────────────────────

def _majority_vote(results: list, field: str):
    """
    Given 3 engine results, return majority-voted value for a field.
    Returns (value, agreement_count).
    """
    values = [r.get(field) for r in results if r.get(field) is not None]
    if not values:
        return None, 0

    # For numeric fields: round to 2dp then vote
    if isinstance(values[0], float):
        rounded = [round(v, 1) for v in values]
        from collections import Counter
        counts = Counter(rounded)
        best_val, best_cnt = counts.most_common(1)[0]
        # Return the original un-rounded value closest to winner
        for v in values:
            if round(v, 1) == best_val:
                return round(v, 2), best_cnt
        return round(best_val, 2), best_cnt

    # For strings: exact match vote
    from collections import Counter
    counts = Counter(values)
    best_val, best_cnt = counts.most_common(1)[0]
    return best_val, best_cnt


def _compute_agreement(results: list) -> tuple:
    """
    Compute overall agreement level across the 3 engines.
    Returns (agreement: str, ocr_confidence: float)
    """
    fields = ["net_subtotal", "vat_amount", "gross_total", "gst_number"]
    agreements = []
    for f in fields:
        values = [r.get(f) for r in results if r.get(f) is not None]
        if len(values) < 2:
            agreements.append(0)
        elif len(values) == 3:
            if values[0] == values[1] == values[2]:
                agreements.append(3)
            elif values[0] == values[1] or values[1] == values[2] or values[0] == values[2]:
                agreements.append(2)
            else:
                agreements.append(1)
        else:
            if values[0] == values[1]:
                agreements.append(2)
            else:
                agreements.append(1)

    avg = sum(agreements) / len(agreements) if agreements else 1

    if avg >= 2.8:
        return "full",     1.00
    elif avg >= 1.8:
        return "majority", 0.67
    else:
        return "split",    0.33


# ── Main Entry Point ─────────────────────────────────────────────────────────

def run_triple_ocr(image_path: str) -> dict:
    """
    Run Tesseract, EasyOCR, and PaddleOCR in parallel.
    Returns majority-voted entities + agreement metadata.

    Output:
        {
            "net_subtotal":   float | None,
            "vat_percent":    float | None,
            "vat_amount":     float | None,
            "gross_total":    float | None,
            "gst_number":     str,
            "invoice_number": str,
            "agreement":      "full" | "majority" | "split",
            "ocr_confidence": float (1.0 / 0.67 / 0.33),
            "engine_results": {
                "tesseract": {...},
                "easyocr":   {...},
                "paddleocr": {...}
            }
        }
    """
    # Run all 3 engines in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "tesseract": executor.submit(_run_tesseract, image_path),
            "easyocr":   executor.submit(_run_easyocr,   image_path),
            "paddleocr": executor.submit(_run_paddleocr,  image_path),
        }
        engine_results = {name: f.result() for name, f in futures.items()}

    results_list = list(engine_results.values())

    # Majority vote on each field
    fields = ["net_subtotal", "vat_percent", "vat_amount", "gross_total",
              "gst_number", "invoice_number"]
    merged = {}
    for f in fields:
        val, cnt = _majority_vote(results_list, f)
        merged[f] = val

    # Agreement level
    agreement, confidence = _compute_agreement(results_list)

    return {
        "net_subtotal":   merged.get("net_subtotal"),
        "vat_percent":    merged.get("vat_percent"),
        "vat_amount":     merged.get("vat_amount"),
        "gross_total":    merged.get("gross_total"),
        "gst_number":     merged.get("gst_number") or "",
        "invoice_number": merged.get("invoice_number") or "",
        "agreement":      agreement,
        "ocr_confidence": confidence,
        "engine_results": engine_results,   # individual engine outputs for UI cards
    }

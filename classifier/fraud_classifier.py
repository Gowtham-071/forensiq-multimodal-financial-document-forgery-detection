"""
classifier/fraud_classifier.py
Adaptive Fusion Classifier — IEEE Contribution 3
Combines visual forensics + OCR text scores with quality-based weighting,
vendor-aware contextual adjustment, and arithmetic semantic hard gate.
"""

import re
import math

# ── GST Validation Regexes ──────────────────────────────────────────────────
GST_INDIAN_RE  = re.compile(
    r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$'
)
GST_GENERIC_RE = re.compile(
    r'^\d{3}-\d{2}-\d{4}$|^\d{2}-\d{2}-\d{4}$'
)


def validate_gst(gst_number: str, bill_format: str = "Indian GST") -> tuple:
    """
    Validate GST / Tax-ID format.
    Returns (is_valid: bool, reason: str)
    """
    if not gst_number:
        return False, "No GST/Tax-ID extracted by OCR"

    g = gst_number.strip().upper()
    if bill_format == "Indian GST":
        if GST_INDIAN_RE.match(g):
            return True, f"Valid Indian GST format: {g}"
        else:
            return False, f"Invalid Indian GST format: {g} (expected: 2-digit state + 10-char PAN + 1Z1)"
    else:
        if GST_GENERIC_RE.match(g):
            return True, f"Valid Tax-ID format: {g}"
        elif len(g) >= 6:
            return True, f"Tax-ID present: {g}"  # lenient for generic
        else:
            return False, f"Tax-ID too short: {g}"


def arithmetic_semantic_gate(net_subtotal, vat_percent, vat_amount,
                              gross_total, tolerance: float = 0.50) -> tuple:
    """
    IEEE Contribution 1 — The Hard Semantic Gate.

    Checks:
      1. |Total - (Net + VAT_amount)| < tolerance
      2. |VAT_amount - Net * VAT% / 100| < tolerance

    Returns (gate_passed: bool, gate_score: float 0-1, reasons: list[str])
    gate_score = 0.0 means perfect consistency; 1.0 means serious fraud
    """
    reasons = []
    failures = 0

    if None in (net_subtotal, vat_amount, gross_total):
        return False, 0.5, ["Incomplete financial fields — cannot verify arithmetic"]

    # Check 1: Total = Net + VAT_amount
    computed_total = round(net_subtotal + vat_amount, 2)
    total_diff     = abs(gross_total - computed_total)
    if total_diff > tolerance:
        reasons.append(
            f"Total mismatch: expected {computed_total:.2f}, found {gross_total:.2f} "
            f"(diff={total_diff:.2f})"
        )
        failures += 1
    else:
        reasons.append(f"✅ Total check passed (diff={total_diff:.2f})")

    # Check 2: VAT_amount = Net * VAT% / 100 (if VAT% available)
    if vat_percent is not None and vat_percent > 0:
        expected_vat = round(net_subtotal * vat_percent / 100, 2)
        vat_diff     = abs(vat_amount - expected_vat)
        if vat_diff > tolerance:
            reasons.append(
                f"VAT mismatch: expected {expected_vat:.2f} at {vat_percent}%, "
                f"found {vat_amount:.2f} (diff={vat_diff:.2f})"
            )
            failures += 1
        else:
            reasons.append(f"✅ VAT check passed (diff={vat_diff:.2f})")

    gate_passed = failures == 0
    gate_score  = min(failures / 2.0, 1.0)  # 0=clean, 0.5=one fail, 1.0=both fail

    return gate_passed, gate_score, reasons


def adaptive_fusion(
    visual_result:  dict,
    ocr_result:     dict,
    vendor_row:     dict | None,
    bill_format:    str = "Indian GST"
) -> dict:
    """
    IEEE Contribution 3 — Adaptive Quality-Based Fusion.

    Args:
        visual_result:  output of run_visual_forensics()
        ocr_result:     output of run_triple_ocr()
        vendor_row:     output of lookup_vendor_by_gst() — None if not enrolled
        bill_format:    'Indian GST' | 'Generic Invoice'

    Returns a complete verdict dict with fraud_score, verdict, evidence, and reasons.
    """
    evidence   = []
    reasons    = []

    # ── A. Quality-based weights (IEEE Contribution 3) ──
    quality = visual_result.get("image_quality_score", 0.5)
    if quality < 0.4:
        w_visual, w_text = 0.45, 0.55
        evidence.append(f"Low image quality ({quality:.2f}) → increased OCR weight")
    else:
        w_visual, w_text = 0.68, 0.32
        evidence.append(f"Good image quality ({quality:.2f}) → standard visual/OCR weights")

    # ── B. Visual score ──
    S_visual = visual_result.get("composite_visual_score", 0.5)
    evidence.append(
        f"Visual score: {S_visual:.3f} "
        f"(ELA={visual_result.get('ela',0):.2f}, "
        f"CNN={visual_result.get('cnn',0):.2f}, "
        f"Noise={visual_result.get('noise',0):.2f})"
    )

    # ── C. OCR / text score ──
    net      = ocr_result.get("net_subtotal")
    vat_pct  = ocr_result.get("vat_percent")
    vat_amt  = ocr_result.get("vat_amount")
    gross    = ocr_result.get("gross_total")
    gst      = ocr_result.get("gst_number", "")
    invoice  = ocr_result.get("invoice_number", "")
    agreement= ocr_result.get("agreement", "split")
    ocr_conf = ocr_result.get("ocr_confidence", 0.33)

    # Arithmetic gate
    gate_passed, gate_score, gate_reasons = arithmetic_semantic_gate(
        net, vat_pct, vat_amt, gross
    )
    reasons.extend(gate_reasons)

    # OCR confidence as a text-fraud signal (low confidence = more suspicious)
    S_text = gate_score * 0.7 + (1 - ocr_conf) * 0.3
    evidence.append(
        f"OCR text score: {S_text:.3f} "
        f"(gate={gate_score:.2f}, agreement={agreement}, confidence={ocr_conf:.2f})"
    )

    # ── D. Base fusion ──
    S_final = w_visual * S_visual + w_text * S_text

    # ── E. Vendor-aware contextual adjustment (IEEE Contribution 4) ──
    vendor_match  = False
    vendor_name   = "Unknown (not enrolled)"
    amount_in_range = True

    # GST format check
    gst_valid, gst_reason = validate_gst(gst, bill_format)
    reasons.append(gst_reason)

    if vendor_row:
        vendor_match = True
        vendor_name  = vendor_row["company_name"]
        S_final     -= 0.12  # registered GST → reduce suspicion
        evidence.append(f"✅ Vendor match: {vendor_name} (GST registered) → score -0.12")

        # Amount range check
        if gross is not None:
            if gross < vendor_row.get("amount_min", 0) or gross > vendor_row.get("amount_max", 9999999):
                S_final      += 0.08
                amount_in_range = False
                evidence.append(
                    f"⚠️ Amount {gross:.2f} outside registered range "
                    f"[{vendor_row['amount_min']}, {vendor_row['amount_max']}] → score +0.08"
                )
            else:
                evidence.append(f"✅ Amount {gross:.2f} within enrolled range")
    else:
        evidence.append(f"ℹ️ GST '{gst}' not in vendor registry — neutral (no penalty)")

    # ── F. Hard gate override ──
    hard_overridden = False
    if not gate_passed:
        S_final = max(S_final, 0.75)
        evidence.append("⛔ Arithmetic gate FAILED — score forced ≥ 0.75")
        hard_overridden = True

    if not gst_valid and not vendor_match:
        S_final = max(S_final, 0.50)
        evidence.append("⚠️ Invalid GST format and not enrolled — score forced ≥ 0.50")

    # ── G. Clamp and verdict ──
    S_final = round(max(0.0, min(1.0, S_final)), 4)

    if S_final < 0.35:
        verdict = "GENUINE"
    elif S_final <= 0.65:
        verdict = "SUSPICIOUS"
    else:
        verdict = "FRAUD"

    return {
        "verdict":         verdict,
        "fraud_score":     S_final,
        "visual_score":    round(S_visual, 4),
        "text_score":      round(S_text, 4),
        "ocr_confidence":  ocr_conf,
        "agreement":       agreement,
        "gate_passed":     gate_passed,
        "hard_overridden": hard_overridden,
        "vendor_match":    vendor_match,
        "vendor_name":     vendor_name,
        "amount_in_range": amount_in_range,
        "gst_number":      gst,
        "gst_valid":       gst_valid,
        "invoice_number":  invoice,
        "net_subtotal":    net,
        "vat_percent":     vat_pct,
        "vat_amount":      vat_amt,
        "gross_total":     gross,
        "quality_score":   round(quality, 4),
        "w_visual":        w_visual,
        "w_text":          w_text,
        "evidence_lines":  evidence,
        "gate_reasons":    reasons,
    }

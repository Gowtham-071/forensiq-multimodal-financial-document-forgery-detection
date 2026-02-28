"""
app.py — FORENSIQ Flask Web Application
Full 4-stream pipeline: Visual Forensics + Triple OCR + Semantic Gate + Vendor Enrollment
Routes: / (verify) · /vendors (enroll) · /history (dashboard) · /metrics (paper figures)
"""

import os
import json
import time
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

# ── Base path (always relative to this file) ──────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
UPLOAD_DIR   = BASE_DIR / "uploads"
REPORTS_DIR  = BASE_DIR / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ── Pipeline imports ───────────────────────────────────────────────────────
from utils.vendor_db       import (lookup_vendor_by_gst, log_bill,
                                    get_all_vendors, enroll_vendor,
                                    get_bill_history, get_dashboard_stats,
                                    get_bills_over_time)
from utils.preprocessing   import preprocess_image
from vision.vision_model   import run_visual_forensics
from ocr.ocr_engine        import run_triple_ocr
from classifier.fraud_classifier import adaptive_fusion

app = Flask(__name__)
app.secret_key = "forensiq_secret_2025"

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


# ── Route 1: Bill Verification ─────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected.", "error")
            return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload JPG, PNG or BMP.", "error")
            return redirect(url_for("index"))

        # Save upload
        img_path = str(UPLOAD_DIR / file.filename)
        file.save(img_path)

        start_time = time.time()

        try:
            # STREAM 1 — Visual Forensics (6 signals)
            visual  = run_visual_forensics(img_path)

            # STREAM 2 — Triple OCR Consensus
            ocr     = run_triple_ocr(img_path)

            # Vendor lookup via extracted GST
            gst     = ocr.get("gst_number", "")
            vendor  = lookup_vendor_by_gst(gst) if gst else None
            bill_fmt= vendor.get("bill_format", "Indian GST") if vendor else "Indian GST"

            # STREAM 3 + 4 — Adaptive Fusion (semantic gate + vendor context)
            fusion  = adaptive_fusion(visual, ocr, vendor, bill_format=bill_fmt)

            time_taken = round(time.time() - start_time, 2)

            # Log to history
            log_bill(
                company_name   = fusion["vendor_name"],
                gst_found      = gst,
                bill_book_id   = fusion.get("invoice_number", ""),
                verdict        = fusion["verdict"],
                fraud_score    = fusion["fraud_score"],
                visual_score   = fusion["visual_score"],
                ocr_confidence = fusion["ocr_confidence"],
                time_taken_sec = time_taken,
            )

            # Make heatmap URL relative for template
            heatmap_path = visual.get("ela_heatmap_path", "")
            heatmap_url  = None
            if heatmap_path and os.path.exists(heatmap_path):
                heatmap_url = "/uploads/heatmaps/" + Path(heatmap_path).name

            result = {
                "verdict":      fusion["verdict"],
                "fraud_score":  fusion["fraud_score"],
                "time_taken":   time_taken,
                "visual":       visual,
                "ocr":          ocr,
                "fusion":       fusion,
                "heatmap_url":  heatmap_url,
                "filename":     file.filename,
            }

        except Exception as e:
            flash(f"Processing error: {str(e)}", "error")
            return redirect(url_for("index"))

    return render_template("index.html", result=result)


# ── Route 2: Vendor Enrollment ─────────────────────────────────────────────

@app.route("/vendors", methods=["GET", "POST"])
def vendors():
    if request.method == "POST":
        company_name = request.form.get("company_name", "").strip()
        gst_raw      = request.form.get("gst_numbers", "").strip()
        amount_min   = request.form.get("amount_min", 0)
        amount_max   = request.form.get("amount_max", 9999999)
        bill_format  = request.form.get("bill_format", "Indian GST")
        notes        = request.form.get("notes", "").strip()

        if not company_name or not gst_raw:
            flash("Company name and at least one GST number are required.", "error")
            return redirect(url_for("vendors"))

        gst_list = [g.strip().upper() for g in gst_raw.replace("\n", ",").split(",") if g.strip()]
        try:
            enroll_vendor(
                company_name = company_name,
                gst_list     = gst_list,
                amount_min   = float(amount_min),
                amount_max   = float(amount_max),
                bill_format  = bill_format,
                notes        = notes,
            )
            flash(f"✅ '{company_name}' enrolled with {len(gst_list)} GST number(s).", "success")
        except Exception as e:
            flash(f"Enrollment error: {str(e)}", "error")

        return redirect(url_for("vendors"))

    all_vendors = get_all_vendors()
    return render_template("vendors.html", vendors=all_vendors)


# ── Route 3: Bill History Dashboard ────────────────────────────────────────

@app.route("/history")
def history():
    bills      = get_bill_history(limit=200)
    stats      = get_dashboard_stats()
    over_time  = get_bills_over_time()
    return render_template("history.html",
                           bills=bills, stats=stats,
                           over_time_json=json.dumps(over_time))


# ── Route 4: Model Metrics (screenshot-ready for IEEE paper) ───────────────

@app.route("/metrics")
def metrics():
    metrics_path = REPORTS_DIR / "metrics.json"
    stats = {}
    if metrics_path.exists():
        with open(str(metrics_path), "r") as f:
            stats = json.load(f)

    # Check which report images exist
    roc_exists     = (REPORTS_DIR / "roc_curve.png").exists()
    cm_exists      = (REPORTS_DIR / "confusion_matrix.png").exists()
    ablation_exists= (REPORTS_DIR / "ablation_table.png").exists()

    return render_template("metrics.html",
                           stats=stats,
                           roc_exists=roc_exists,
                           cm_exists=cm_exists,
                           ablation_exists=ablation_exists)


# ── Static: serve heatmaps and reports ────────────────────────────────────

@app.route("/uploads/heatmaps/<filename>")
def heatmap_file(filename):
    return send_from_directory(str(UPLOAD_DIR / "heatmaps"), filename)


@app.route("/reports/<filename>")
def report_file(filename):
    return send_from_directory(str(REPORTS_DIR), filename)


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

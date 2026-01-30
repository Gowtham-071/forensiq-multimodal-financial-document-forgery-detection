from flask import Flask, render_template, request
import os
import cv2
import pytesseract
import re
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = r"G:\fraud_document_ai\models\fraud_document_cnn.h5"
model = load_model(MODEL_PATH)

IMG_SIZE = 128


# ---------------- OCR & ENTITY EXTRACTION ----------------

def normalize_numbers(text):
    text = text.replace(" ", "")
    text = text.replace(",", ".")
    return text


def extract_entities(image_path):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    text = pytesseract.image_to_string(Image.fromarray(img_rgb))
    norm_text = normalize_numbers(text)

    # Isolate summary
    if "SUMMARY" not in norm_text:
        return None

    summary_text = norm_text[norm_text.find("SUMMARY"):]
    lines = summary_text.split("\n")

    # VAT %
    vat_match = re.search(r'(\d+)%', summary_text)
    vat_percent = float(vat_match.group(1)) if vat_match else None

    # Total line
    total_line = [l for l in lines if "Total" in l]
    if not total_line:
        return None

    numbers = re.findall(r'\d+\.\d+', total_line[0])
    if len(numbers) < 3:
        return None

    net_subtotal = float(numbers[0])
    vat_amount = float(numbers[1])
    gross_total = float(numbers[2])

    return net_subtotal, vat_percent, vat_amount, gross_total


# ---------------- NUMERIC FRAUD CHECK ----------------

def numeric_fraud_check(net_subtotal, vat_percent, vat_amount, gross_total, tol=1.0):
    reasons = []

    computed_vat = round(net_subtotal * vat_percent / 100, 2)
    computed_total = round(net_subtotal + computed_vat, 2)

    if abs(vat_amount - computed_vat) > tol:
        reasons.append(f"VAT mismatch (expected {computed_vat}, found {vat_amount})")

    if abs(gross_total - computed_total) > tol:
        reasons.append(f"Total mismatch (expected {computed_total}, found {gross_total})")

    if reasons:
        return "FRAUD", reasons
    else:
        return "GENUINE", ["All financial values are consistent"]


# ---------------- CNN VISUAL RISK ----------------

def preprocess_for_cnn(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    roi = img[int(h * 0.7):h, 0:w]  # bottom ROI
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi / 255.0
    roi = roi.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return roi


def interpret_cnn_score(score):
    if score >= 0.65:
        return "High visual tampering risk"
    elif score >= 0.40:
        return "Moderate visual anomalies detected"
    else:
        return "Low visual tampering risk"


# ---------------- FLASK ROUTE ----------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    reasons = []
    cnn_score = None
    cnn_reason = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            entities = extract_entities(path)

            if entities:
                net_sub, vat_p, vat_amt, gross = entities
                result, reasons = numeric_fraud_check(
                    net_sub, vat_p, vat_amt, gross
                )
            else:
                result = "SUSPICIOUS"
                reasons = ["Unable to extract financial entities"]

            # CNN visual evidence
            roi = preprocess_for_cnn(path)
            cnn_score = float(model.predict(roi)[0][0])
            cnn_reason = interpret_cnn_score(cnn_score)

    return render_template(
        "index.html",
        result=result,
        reasons=reasons,
        cnn_score=cnn_score,
        cnn_reason=cnn_reason
    )


if __name__ == "__main__":
    app.run(debug=True)

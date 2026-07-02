"""
Project 4 - Web App Server
DecodeLabs | Batch 2026
Features: OCR, Object Detection, Webcam, Multi-Image, PDF Report
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
import pytesseract
import cv2
import numpy as np
import os
import base64
from PIL import Image
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import json
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__, static_folder='static')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

PROTO = "models/MobileNetSSD_deploy.prototxt"
MODEL = "models/MobileNetSSD_deploy.caffemodel"

# Store results for PDF
last_results = {}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/ocr', methods=['POST'])
def ocr():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh, config="--oem 3 --psm 6")
    data = pytesseract.image_to_data(thresh, config="--oem 3 --psm 6",
                                      output_type=pytesseract.Output.DICT)

    confident = [data["text"][i] for i in range(len(data["conf"]))
                 if data["text"][i].strip() and str(data["conf"][i]) != '-1'
                 and int(data["conf"][i]) >= 80]

    confs = [int(c) for c in data["conf"] if str(c) != '-1' and int(c) > 0]
    avg_conf = sum(confs) / len(confs) if confs else 0

    result = {
        "type": "OCR",
        "text": text.strip(),
        "words": len(confident),
        "confidence": round(avg_conf, 1),
        "passed": avg_conf >= 80,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    last_results.update(result)
    return jsonify(result)

@app.route('/detect', methods=['POST'])
def detect():
    if not os.path.exists(PROTO) or not os.path.exists(MODEL):
        return jsonify({"error": "Model files not found!"}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    (h, w) = img_cv.shape[:2]

    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
    blob = cv2.dnn.blobFromImage(cv2.resize(img_cv, (300, 300)),
                                  0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    results = []
    output = img_cv.copy()

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= 0.50:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            label = CLASSES[idx] if idx < len(CLASSES) else "unknown"
            results.append({"label": label, "confidence": round(conf * 100, 1)})
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 120, 255), 3)
            cv2.putText(output, f"{label}: {round(conf*100,1)}%",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)

    _, buffer = cv2.imencode('.png', output)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    result = {
        "type": "Detection",
        "objects": results,
        "total": len(results),
        "image": img_base64,
        "passed": len(results) > 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    last_results.update(result)
    return jsonify(result)

@app.route('/ocr_multiple', methods=['POST'])
def ocr_multiple():
    files = request.files.getlist('images')
    all_results = []

    for file in files:
        img = Image.open(file.stream).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config="--oem 3 --psm 6")
        data = pytesseract.image_to_data(thresh, config="--oem 3 --psm 6",
                                          output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data["conf"] if str(c) != '-1' and int(c) > 0]
        avg_conf = sum(confs) / len(confs) if confs else 0

        all_results.append({
            "filename": file.filename,
            "text": text.strip(),
            "confidence": round(avg_conf, 1),
            "passed": avg_conf >= 80
        })

    return jsonify({"results": all_results})

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    data = request.json
    pdf_path = "report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("DecodeLabs AI Vision Engine", styles['Title']))
    story.append(Paragraph("Project 4 — Batch 2026", styles['Normal']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Results
    if data.get('type') == 'OCR':
        story.append(Paragraph("OCR Results", styles['Heading1']))
        story.append(Spacer(1, 10))
        table_data = [
            ['Words Detected', str(data.get('words', 0))],
            ['Confidence Score', f"{data.get('confidence', 0)}%"],
            ['Milestone', 'PASSED ✓' if data.get('passed') else 'FAILED ✗'],
        ]
        t = Table(table_data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        story.append(Paragraph("Extracted Text:", styles['Heading2']))
        story.append(Paragraph(data.get('text', '').replace('\n', '<br/>'), styles['Normal']))

    elif data.get('type') == 'Detection':
        story.append(Paragraph("Object Detection Results", styles['Heading1']))
        story.append(Spacer(1, 10))
        table_data = [['Object', 'Confidence']]
        for obj in data.get('objects', []):
            table_data.append([obj['label'].capitalize(), f"{obj['confidence']}%"])
        t = Table(table_data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t)

    doc.build(story)

    # Read PDF into memory and send
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()

    from flask import Response
    return Response(
        pdf_data,
        mimetype='application/pdf',
        headers={
            'Content-Disposition': 'attachment; filename=DecodeLabs_Report.pdf',
            'Content-Type': 'application/pdf',
            'Content-Length': len(pdf_data)
        }
    )

if __name__ == '__main__':
    print("\n=======================================")
    print("  DecodeLabs AI Vision Engine")
    print("  Project 4 | Batch 2026")
    print("=======================================")
    print("\n  🚀 Server is RUNNING!")
    print("=======================================\n")
    app.run(debug=False, port=5000)
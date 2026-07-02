# DecodeLabs AI Vision Engine — Project 4

**Industrial Training Kit | Batch 2026 | Powered by DecodeLabs**

An AI-powered Image & Text Recognition web application built with Python and Flask.

---
*Built & Developed by Sumera
Internship | Batch 2026*

## What This Project Does

This project gives a machine the ability to read text from images (OCR) and detect objects in photos (Object Detection) — all through a professional web dashboard running locally.

---

## Features

- OCR — extract text from any image
- Object Detection — detect and label objects using MobileNet-SSD
- Live Webcam Scan — snap and scan directly from camera
- Batch Scan — upload and scan multiple images at once
- Scan History — all past results saved automatically
- Dashboard — charts showing confidence scores and scan statistics
- PDF Report — download scan results as a professional PDF
- Copy Text — one-click copy of extracted OCR text

---

## Tech Stack

- Python 3.10+
- Flask — web server
- OpenCV — image processing & object detection
- pytesseract — OCR text extraction
- MobileNet-SSD — pre-trained object detection model
- ReportLab — PDF generation
- Chart.js — dashboard charts

---

## Project Structure

```
Project 4 Decode/
├── ocr/
│   └── ocr_engine.py
├── object_detection/
│   └── object_detection.py
├── models/
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
├── static/
│   └── index.html
├── server.py
├── requirements.txt
└── README.md
```

---

## Setup

**Step 1 — Activate virtual environment:**
```
.\venv\Scripts\activate
```

**Step 2 — Install dependencies:**
```
pip install -r requirements.txt
```

**Step 3 — Install Tesseract OCR:**
Download from:
```
https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe
```

**Step 4 — Run the app:**
```
python server.py
```

**Step 5 — Open in browser:**
```
http://localhost:5000
```

---

## Milestone Results

| Milestone | Status |
|---|---|
| Library Integration | Passed |
| Pre-Processing Integrity | Passed |
| Accuracy Benchmarking | Passed — OCR: 95.4% · Detection: 87.1% |
| Visual Confirmation | Passed |

---

## Troubleshooting

| Error | Fix |
|---|---|
| TesseractNotFoundError | Install Tesseract (Step 3 above) |
| ModuleNotFoundError | Run pip install -r requirements.txt |
| Model files not found | Place .prototxt and .caffemodel in models/ folder |

---

*Sumera | Computer Systems — 3rd Year | DecodeLabs Internship | Batch 2026*

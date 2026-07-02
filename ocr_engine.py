"""
Project 4 - Path 1: OCR
DecodeLabs | Batch 2026
"""

import cv2
import pytesseract
import numpy as np
import os
import sys
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def run_ocr_pipeline(image_path, psm_mode=3):
    print("\n=======================================")
    print("  PROJECT 4 — OCR Engine")
    print("  DecodeLabs | Batch 2026")
    print("=======================================\n")

    # Load image
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return
    image = cv2.imread(image_path)
    print(f"[✓] Image loaded: {image_path}")

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("[✓] Grayscale conversion done")

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("[✓] Gaussian blur applied")

    # Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("[✓] Adaptive thresholding done")

    # Save preprocessed
    cv2.imwrite("preprocessed.png", thresh)
    print("[✓] Preprocessed image saved → preprocessed.png")

    # OCR
    config = f"--oem 3 --psm {psm_mode}"
    text = pytesseract.image_to_string(thresh, config=config)
    data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

    # Filter confident words
    confident_words = []
    for i, conf in enumerate(data["conf"]):
        try:
            if int(conf) >= 80:
                word = data["text"][i].strip()
                if word:
                    confident_words.append({
                        "word": word,
                        "confidence": int(conf),
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "w": data["width"][i],
                        "h": data["height"][i],
                    })
        except:
            continue

    # Draw boxes
    output = image.copy()
    for item in confident_words:
        x, y, w, h = item["x"], item["y"], item["w"], item["h"]
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 200, 0), 2)
        cv2.putText(output, f"{item['confidence']}%", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    cv2.imwrite("ocr_output.png", output)
    print("[✓] Output image saved → ocr_output.png")

    # Results
    avg_conf = sum(w["confidence"] for w in confident_words) / len(confident_words) if confident_words else 0

    print("\n=======================================")
    print("  EXTRACTED TEXT:")
    print("=======================================")
    for line in text.splitlines():
        if line.strip():
            print(f"  {line}")
    print(f"\n  Words detected: {len(confident_words)}")
    print(f"  Average confidence: {avg_conf:.1f}%")
    print("=======================================")

    if avg_conf >= 80:
        print("  [✓] MILESTONE PASSED: Confidence >= 80%")
    else:
        print("  [!] Low confidence — try a clearer image")
    print("=======================================\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_engine.py <image> [psm]")
        sys.exit(1)
    img = sys.argv[1]
    psm = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run_ocr_pipeline(img, psm)
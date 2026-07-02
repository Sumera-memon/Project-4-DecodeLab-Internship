"""
Project 4 - Path 2: Object Detection
DecodeLabs | Batch 2026
"""

import cv2
import numpy as np
import os
import sys

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

PROTO = "models/MobileNetSSD_deploy.prototxt"
MODEL = "models/MobileNetSSD_deploy.caffemodel"

def run_detection(image_path, confidence_threshold=0.80):
    print("\n=======================================")
    print("  PROJECT 4 — Object Detection")
    print("  DecodeLabs | Batch 2026")
    print("=======================================\n")

    # Check model files
    if not os.path.exists(PROTO) or not os.path.exists(MODEL):
        print("[ERROR] Model files not found in models/ folder!")
        print("  Need: MobileNetSSD_deploy.prototxt")
        print("  Need: MobileNetSSD_deploy.caffemodel")
        return

    # Load image
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    print(f"[✓] Image loaded: {image_path}")

    # Load network
    print("[→] Loading MobileNet-SSD network...")
    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
    print("[✓] Network loaded!")

    # Create blob
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007843, (300, 300), 127.5
    )
    print("[→] Running detection...")
    net.setInput(blob)
    detections = net.forward()

    # Process detections
    results = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            label = f"{CLASSES[idx]}: {round(conf*100, 1)}%"
            results.append({"label": CLASSES[idx], "confidence": round(conf*100, 1)})

            # Draw box
            color = COLORS[idx]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save output
    cv2.imwrite("detection_output.png", image)
    print("[✓] Output saved → detection_output.png")

    # Results
    print("\n=======================================")
    print("  DETECTED OBJECTS:")
    print("=======================================")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['label']} — {r['confidence']}%")
    print(f"\n  Total: {len(results)} objects")
    print("=======================================")
    if results:
        print("  [✓] MILESTONE PASSED: Confidence >= 80%")
    else:
        print("  [!] No objects found — try clearer image")
    print("=======================================\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python object_detection.py <image> [confidence]")
        sys.exit(1)
    img = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.80
    run_detection(img, conf)
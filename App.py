"""
Project 4 - GUI App
DecodeLabs | Batch 2026
Click and scan any image!
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pytesseract
import cv2
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ── Setup Main Window ──
root = tk.Tk()
root.title("DecodeLabs — Project 4 | AI Vision Engine")
root.geometry("900x650")
root.configure(bg="#0D1117")

# ── Title ──
title = tk.Label(
    root,
    text="🔬 DecodeLabs AI Vision Engine",
    font=("Arial", 22, "bold"),
    bg="#0D1117", fg="white"
)
title.pack(pady=20)

subtitle = tk.Label(
    root,
    text="Project 4 | Batch 2026 | OCR + Object Detection",
    font=("Arial", 11),
    bg="#0D1117", fg="#6B7280"
)
subtitle.pack()

# ── Image Display ──
img_frame = tk.Frame(root, bg="#161B22", width=500, height=300)
img_frame.pack(pady=20)
img_frame.pack_propagate(False)

img_label = tk.Label(img_frame, bg="#161B22", text="No image selected", fg="#6B7280", font=("Arial", 12))
img_label.pack(expand=True)

# ── Result Box ──
result_label = tk.Label(root, text="RESULTS:", font=("Arial", 11, "bold"), bg="#0D1117", fg="white")
result_label.pack()

result_box = tk.Text(
    root, height=8, width=80,
    font=("Courier", 11),
    bg="#161B22", fg="#00FF88",
    relief="flat", padx=10, pady=10
)
result_box.pack(pady=5)

# ── Status Bar ──
status = tk.Label(root, text="Ready", font=("Arial", 10), bg="#0D1117", fg="#6B7280")
status.pack()

# ── Functions ──
selected_image_path = None

def show_image(path):
    img = Image.open(path)
    img.thumbnail((480, 280))
    photo = ImageTk.PhotoImage(img)
    img_label.configure(image=photo, text="")
    img_label.image = photo

def run_ocr():
    global selected_image_path
    if not selected_image_path:
        messagebox.showwarning("No Image", "Please select an image first!")
        return

    status.config(text="Running OCR...", fg="yellow")
    root.update()

    # Pre-process
    image = cv2.imread(selected_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR
    text = pytesseract.image_to_string(thresh, config="--oem 3 --psm 6")
    data = pytesseract.image_to_data(thresh, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT)

    confident = [data["text"][i] for i in range(len(data["conf"]))
                 if data["text"][i].strip() and int(data["conf"][i]) >= 80]

    avg_conf = sum([int(c) for c in data["conf"] if c != "-1"]) / len(data["conf"])

    result_box.delete(1.0, tk.END)
    result_box.insert(tk.END, f"📄 EXTRACTED TEXT:\n")
    result_box.insert(tk.END, f"{'='*50}\n")
    result_box.insert(tk.END, text)
    result_box.insert(tk.END, f"\n{'='*50}\n")
    result_box.insert(tk.END, f"✅ Words detected: {len(confident)}\n")
    result_box.insert(tk.END, f"📊 Confidence: {avg_conf:.1f}%\n")

    if avg_conf >= 80:
        result_box.insert(tk.END, "🏆 MILESTONE PASSED!\n")
        status.config(text="✅ OCR Complete — Milestone Passed!", fg="#00FF88")
    else:
        status.config(text="⚠️ Low confidence — try clearer image", fg="orange")

def run_detection():
    global selected_image_path
    if not selected_image_path:
        messagebox.showwarning("No Image", "Please select an image first!")
        return

    PROTO = "models/MobileNetSSD_deploy.prototxt"
    MODEL = "models/MobileNetSSD_deploy.caffemodel"

    if not os.path.exists(PROTO) or not os.path.exists(MODEL):
        messagebox.showerror("Error", "Model files not found in models/ folder!")
        return

    status.config(text="Running Detection...", fg="yellow")
    root.update()

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    image = cv2.imread(selected_image_path)
    (h, w) = image.shape[:2]

    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= 0.50:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            label = f"{CLASSES[idx]}: {round(conf*100,1)}%"
            results.append({"label": CLASSES[idx], "confidence": round(conf*100,1)})
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    output_path = "detection_output.png"
    cv2.imwrite(output_path, image)
    show_image(output_path)

    result_box.delete(1.0, tk.END)
    result_box.insert(tk.END, f"🔍 DETECTED OBJECTS:\n")
    result_box.insert(tk.END, f"{'='*50}\n")
    for i, r in enumerate(results, 1):
        result_box.insert(tk.END, f"{i}. {r['label']} — {r['confidence']}%\n")
    result_box.insert(tk.END, f"{'='*50}\n")
    result_box.insert(tk.END, f"Total objects: {len(results)}\n")

    if results:
        result_box.insert(tk.END, "🏆 MILESTONE PASSED!\n")
        status.config(text="✅ Detection Complete — Milestone Passed!", fg="#00FF88")
    else:
        status.config(text="⚠️ No objects found — try different image", fg="orange")

def select_image():
    global selected_image_path
    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if path:
        selected_image_path = path
        show_image(path)
        status.config(text=f"Image loaded: {os.path.basename(path)}", fg="white")
        result_box.delete(1.0, tk.END)

# ── Buttons ──
btn_frame = tk.Frame(root, bg="#0D1117")
btn_frame.pack(pady=10)

btn_select = tk.Button(
    btn_frame, text="📁 Choose Image",
    font=("Arial", 12, "bold"),
    bg="#2563EB", fg="white",
    padx=20, pady=8, relief="flat",
    cursor="hand2", command=select_image
)
btn_select.grid(row=0, column=0, padx=10)

btn_ocr = tk.Button(
    btn_frame, text="🔍 Run OCR",
    font=("Arial", 12, "bold"),
    bg="#16A34A", fg="white",
    padx=20, pady=8, relief="flat",
    cursor="hand2", command=run_ocr
)
btn_ocr.grid(row=0, column=1, padx=10)

btn_detect = tk.Button(
    btn_frame, text="📦 Detect Objects",
    font=("Arial", 12, "bold"),
    bg="#C0392B", fg="white",
    padx=20, pady=8, relief="flat",
    cursor="hand2", command=run_detection
)
btn_detect.grid(row=0, column=2, padx=10)

# ── Run ──
root.mainloop()
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime
import os

# === CONFIG ===
TARGET_IMAGE = 'target.jpg'
LOG_FILE = 'logs/detections.csv'
VIDEO_SOURCE = 0  # Use camera index or path to video file
TOLERANCE = 0.25  # Lower = stricter face match

# === SETUP ===
os.makedirs("logs", exist_ok=True)
model = YOLO('yolov8n.pt')  # Or yolov8m.pt for better accuracy

# Embed target image
print("🔬 Encoding target face...")
target_repr = DeepFace.represent(img_path=TARGET_IMAGE, model_name='Facenet')[0]['embedding']

# Load past logs if they exist
if os.path.exists(LOG_FILE):
    detections = pd.read_csv(LOG_FILE)
else:
    detections = pd.DataFrame(columns=['timestamp', 'match'])

# Start video
cap = cv2.VideoCapture(VIDEO_SOURCE)
print("🎥 Video stream started. Scanning for match...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        if name == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]

            try:
                probe = DeepFace.represent(img=face_crop, img_path=None, model_name='Facenet', enforce_detection=False)[0]['embedding']
                dist = np.linalg.norm(np.array(probe) - np.array(target_repr))
                if dist < TOLERANCE:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    detections.loc[len(detections)] = [timestamp, 'MATCH']
                    detections.to_csv(LOG_FILE, index=False)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, "MATCH FOUND", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            except Exception as e:
                print("Error during match:", e)

    cv2.imshow("🦾 ArcScanner - Face Search", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Scan complete. Logged detections saved to 'logs/detections.csv'")
from datetime import datetime
import cv2
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db

# Firebase initialize
if not firebase_admin._apps:
  cred = credentials.Certificate("serviceAccountKey.json")
  firebase_admin.initialize_app(
      cred,
      {"databaseURL": "https://license-plate-detection-anpr-default-rtdb.firebaseio.com/"},
  )

# Model load
model = YOLO("runs/detect/license_plate_run-6/weights/best.pt")

# Video path
video_path = "License Plate Detection Test_720p.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0  # <--- Yahan frame counter shuru kiya hai
print("Live video stream simulation shuru ho gayi hai...")

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    print("Video khatam ho gayi hai.")
    break

  frame_count += 1  # <--- Har naye frame par 1 plus ho jaye ga

  # Model inference
  results = model(frame, conf=0.4)

  for r in results:
    boxes = r.boxes
    if len(boxes) > 0:
      detections_list = []

      for box in boxes:
        coords = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections_list.append(
            {
                "confidence": round(conf, 2),
                "bbox": [
                    round(coords[0], 1),
                    round(coords[1], 1),
                    round(coords[2], 1),
                    round(coords[3], 1),
                ],
            }
        )

      # Firebase data packet mein frame_number bhi add kar diya hai
      data = {
          "camera_id": "Main_Gate_Cam",
          "frame_number": frame_count,  # <--- Yeh bataye ga kaunsa frame hai
          "timestamp": str(datetime.now()),
          "total_plates": len(detections_list),
          "detections": detections_list,
      }

      ref = db.reference("license_plates")
      new_ref = ref.push(data)
      print(
          f"Frame #{frame_count} par Plate Detect Hui! Firebase Key:"
          f" {new_ref.key}"
      )

cap.release()
print("Simulated live stream mukammal ho gayi!")
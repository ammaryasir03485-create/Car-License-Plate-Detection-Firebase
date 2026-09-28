from ultralytics import YOLO

# Model load karein jo aapne save kiya hai (ya last.pt se)
model = YOLO("runs/detect/license_plate_run-6/weights/last.pt")

# Yahan epochs=50 define kar dein
model.train(resume=True, epochs=50) 
from ultralytics import YOLO

# Apne trained best weights ka path load karein
model = YOLO("C:/Users/NICAT/Desktop/plate_detection/runs/detect/license_plate_run-6/weights/best.pt")

# Test images ya folder ka path source mein dein
results = model.predict(source="C:/Users/NICAT/Desktop/plate_detection/test_images", conf=0.25, show=True, save=True)
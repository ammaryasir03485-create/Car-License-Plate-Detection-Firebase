from ultralytics import YOLO

# Model load karein (chote ya balanced objects ke liye yolov8s use karna behtar hai)
model = YOLO('yolov8s.pt')

# Training script with optimized settings for small number plates
results = model.train(
    data=r"C:/Users/NICAT/Desktop/plate_detection/data.yaml",
    epochs=50,
    imgsz=800,     # Choti objects ke liye ap isay 800 ya 1024 bhi kar sakte hain
    batch=16,     # Apne GPU memory ke hisab se batch size adjust karein
    workers=0,
    name='license_plate_run',
    device=0
)

print("License plate model ki training successfully start ho chuki hai!")
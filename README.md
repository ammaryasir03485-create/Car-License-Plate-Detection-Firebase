# Real-Time Car License Plate Detection & Firebase Telemetry Sync

This repository features an advanced computer vision system built to detect vehicle license plates using **Ultralytics YOLOv8** and instantly synchronize vehicle detection logs, timestamps, and telemetry data to a **Google Firebase Realtime Database**.

---

## 🚀 Project Overview
Automated Traffic Management and Smart Parking systems require instant data logging. This system detects vehicles and their license plates in real-time video streams, extracts the plate regions, and pushes live logs directly to the cloud via Firebase for remote monitoring and telemetry sync.

---

## 🛠️ Tech Stack & Libraries
* **Programming Language:** Python
* **Object Detection:** Ultralytics YOLOv8
* **Cloud Database:** Google Firebase Realtime Database (`firebase_admin`)
* **Image Processing:** OpenCV (`cv2`), NumPy, Pandas

---

## 📋 Key Features
* **Plate Localization:** High-accuracy YOLOv8 bounding box detection specifically trained for license plates.
* **Cloud Telemetry Sync:** Automatically logs detection events, timestamps, and metadata to Firebase in real-time (`sync_database.py`).
* **Frame Extraction & Preprocessing:** Dedicated scripts for handling video frame extractions and preprocessing.
* **Model Weights:** Includes trained `.pt` weight files (`best.pt`) optimized for detection tasks.

---

## 📂 Repository Structure
```text
Car-License-Plate-Detection-Firebase/
│
├── best.pt                 # Trained YOLOv8 license plate weights
├── sync_database.py        # Main detection & Firebase sync script
├── target_frame.py         # Frame extraction utility script
├── data                    # Dataset configuration file
└── README.md               # Project documentation

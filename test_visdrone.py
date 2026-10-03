from ultralytics import YOLO
import cv2
import os

MODEL_PATH = r"C:\Users\user\yolo11\runs\detect\runs\detect\visdrone_yolo11s_50ep\weights\best.pt"
VIDEO_PATH = r"C:\Users\user\yolo11\test_video.mp4"

CONFIDENCE = 0.35
IMGSZ = 640

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

print("=" * 60)
print("VISDRONE CLEAN DETECTION TEST")
print("=" * 60)
print("Model:", MODEL_PATH)
print("Confidence:", CONFIDENCE)
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        conf=CONFIDENCE,
        imgsz=IMGSZ,
        verbose=False
    )

    result = results[0]

    annotated = result.plot(
        boxes=True,
        labels=True,
        conf=True
    )

    cv2.imshow("VISDRONE CLEAN TEST", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
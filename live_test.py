import cv2
import os
import time
import torch
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "runs",
    "detect",
    "visdrone_yolo11s_50ep",
    "weights",
    "best.pt"
)

# DroidCam only
DROIDCAM_INDEX = 1

# Try Windows camera backends in this order
CAMERA_BACKENDS = [
    ("MSMF", cv2.CAP_MSMF),
    ("DSHOW", cv2.CAP_DSHOW),
    ("ANY", cv2.CAP_ANY),
]

CONFIDENCE = 0.15
IOU = 0.50
IMG_SIZE = 640

# VisDrone classes:
# 0 = pedestrian
# 1 = people
PERSON_CLASSES = [0, 1]

WINDOW_NAME = "LIVE BEST.PT TEST - DROIDCAM"


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("LIVE BEST.PT TEST - DROIDCAM")
print("=" * 60)


# ============================================================
# CUDA CHECK
# ============================================================

print()
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    DEVICE = 0
else:
    print("WARNING: CUDA unavailable - using CPU")
    DEVICE = "cpu"


# ============================================================
# MODEL
# ============================================================

if not os.path.isfile(MODEL_PATH):
    print()
    print("ERROR: best.pt not found!")
    print("Expected:")
    print(MODEL_PATH)
    input("Press ENTER...")
    raise SystemExit


print()
print("Loading trained YOLO11 model...")
print("Model:", MODEL_PATH)

try:
    model = YOLO(MODEL_PATH)

    if torch.cuda.is_available():
        model.to("cuda:0")

    print("Model loaded successfully.")

except Exception as e:
    print()
    print("ERROR: MODEL LOAD FAILED")
    print(e)
    input("Press ENTER...")
    raise SystemExit


# ============================================================
# CAMERA
# ============================================================

print()
print("Opening DroidCam...")
print("Camera index:", DROIDCAM_INDEX)

cap = None
backend_used = None

for backend_name, backend in CAMERA_BACKENDS:

    print(f"Trying DroidCam index {DROIDCAM_INDEX} with {backend_name}...")

    test_cap = cv2.VideoCapture(DROIDCAM_INDEX, backend)

    if test_cap.isOpened():

        # Request reasonable resolution
        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Give camera a moment
        time.sleep(0.5)

        ret, frame = test_cap.read()

        if ret and frame is not None:
            cap = test_cap
            backend_used = backend_name
            print(f"Camera opened successfully using {backend_name}")
            break

        test_cap.release()

    else:
        test_cap.release()


if cap is None:
    print()
    print("ERROR: Could not open DroidCam.")
    print()
    print("Make sure:")
    print("1. DroidCam Client is running on the laptop")
    print("2. DroidCam phone app is running")
    print("3. USB/Wi-Fi connection is active")
    print("4. DroidCam is camera index 1")
    print()
    input("Press ENTER...")
    raise SystemExit


# ============================================================
# CAMERA SETTINGS
# ============================================================

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print()
print("DroidCam backend:", backend_used)
print("Camera ready.")
print()
print("CONTROLS")
print("  Q = quit")
print("  SPACE = pause")
print()


# ============================================================
# FPS
# ============================================================

fps = 0.0
fps_counter = 0
fps_start = time.time()

frame_number = 0
paused = False

last_results = None


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    if not paused:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("WARNING: Failed to read DroidCam frame")
            time.sleep(0.05)
            continue

        frame_number += 1

        # ----------------------------------------------------
        # YOLO DETECTION
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # classes=[0, 1]
        #
        # 0 = pedestrian
        # 1 = people
        #
        # This prevents the model from trying to display
        # cars, bicycles, trucks, etc. during this test.
        # ----------------------------------------------------

        results = model.predict(
            frame,
            imgsz=640,
            conf=0.15,
            iou=0.50,
            classes=[0, 1],
            device=DEVICE,
            verbose=False
        )

        last_results = results

        # ----------------------------------------------------
        # DRAW DETECTIONS
        # ----------------------------------------------------

        people_count = 0

        if results and len(results) > 0:

            result = results[0]

            if result.boxes is not None:

                boxes = result.boxes

                for box in boxes:

                    # Confidence
                    confidence = float(box.conf[0])

                    # Class
                    class_id = int(box.cls[0])

                    # Only people
                    if class_id not in PERSON_CLASSES:
                        continue

                    people_count += 1

                    # Bounding box
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    # Class name
                    if class_id == 0:
                        label = "PEDESTRIAN"
                    else:
                        label = "PEOPLE"

                    label_text = f"{label} {confidence:.2f}"

                    # ------------------------------------------------
                    # Bounding box
                    # ------------------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # ------------------------------------------------
                    # Label background
                    # ------------------------------------------------

                    (tw, th), _ = cv2.getTextSize(
                        label_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        2
                    )

                    label_y = max(y1 - 8, th + 5)

                    cv2.rectangle(
                        frame,
                        (x1, label_y - th - 8),
                        (x1 + tw + 8, label_y),
                        (0, 255, 0),
                        -1
                    )

                    # ------------------------------------------------
                    # Label
                    # ------------------------------------------------

                    cv2.putText(
                        frame,
                        label_text,
                        (x1 + 4, label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA
                    )


        # ========================================================
        # FPS
        # ========================================================

        fps_counter += 1

        elapsed = time.time() - fps_start

        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()


        # ========================================================
        # HUD
        # ========================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (440, 115),
            (0, 0, 0),
            -1
        )

        cv2.rectangle(
            frame,
            (10, 10),
            (440, 115),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "YOLO11 BEST.PT | PEOPLE TEST",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"PEOPLE: {people_count}",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"CONF: {CONFIDENCE:.2f} | FRAME: {frame_number}",
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # ========================================================
        # TOP RIGHT
        # ========================================================

        fps_text = f"{fps:.1f} FPS"

        cv2.putText(
            frame,
            fps_text,
            (
                frame.shape[1] - 150,
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # ========================================================
        # BOTTOM
        # ========================================================

        cv2.putText(
            frame,
            "Q: QUIT   SPACE: PAUSE",
            (25, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


    # ============================================================
    # PAUSED SCREEN
    # ============================================================

    else:

        if last_results is not None:
            pass

        cv2.putText(
            frame,
            "PAUSED",
            (
                frame.shape[1] // 2 - 80,
                frame.shape[0] // 2
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3,
            cv2.LINE_AA
        )


    # ============================================================
    # DISPLAY
    # ============================================================

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == 32:
        paused = not paused

        if paused:
            print("PAUSED")
        else:
            print("RESUMED")


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("LIVE TEST FINISHED")
print("=" * 60)
print(f"Frames processed: {frame_number}")
print(f"Average/latest FPS: {fps:.1f}")
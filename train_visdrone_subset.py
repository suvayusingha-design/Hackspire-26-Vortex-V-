from ultralytics import YOLO

print("=" * 60)
print("VISDRONE SUBSET TRAINING")
print("=" * 60)

model = YOLO("yolo11n.pt")

results = model.train(
    data=r"D:\Project\yolo11\datasets\VisDrone_subset\data.yaml",

    # Training
    epochs=5,
    imgsz=640,
    batch=1,

    # CPU
    device="cpu",
    workers=0,

    # Don't cache the images in RAM
    cache=False,

    # Start from pretrained YOLO11n
    pretrained=True,

    # Stop if there is no improvement
    patience=3,

    # Reduce unnecessary augmentation for the first experiment
    degrees=0.0,
    translate=0.10,
    scale=0.50,
    fliplr=0.50,

    # Output
    project=r"D:\Project\yolo11\runs\detect",
    name="visdrone_subset_5ep",

    verbose=True
)

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print()
print("Best model:")
print(r"D:\Project\yolo11\runs\detect\visdrone_subset_5ep\weights\best.pt")
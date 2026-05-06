from ultralytics import YOLO

model = YOLO("runs/detect/skipper/skipper_detector-2/weights/last.pt")

model.train(
    data="skipper.yaml",
    name="skipper_detector",
    project="skipper",
    imgsz=640,
    batch=4,
    epochs=30,
    patience=15,
)

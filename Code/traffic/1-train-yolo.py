'''
This code trains a YOLO model on a set of images and finally exports it.

'''
from ultralytics import YOLO
import yaml

def main():
    with open("Code/traffic/detrac.yaml") as file:
        config = yaml.safe_load(file)

    model = YOLO("yolov8s.pt")

    model.train(
        data="Code/traffic/detrac.yaml",
        epochs=10,
        batch=16,
        imgsz=640,
        device=0
    )

    metrics = model.val()
    print(metrics)

    results = model.predict(
        source=config["test"],
        conf=0.25,
        save=True
    )

    model.export(format="onnx")

# WHEN USING DEVICE=GPU THE CODE MUST BE INSIDE OF IF NAME MAIN
if __name__ == "__main__":
    main()

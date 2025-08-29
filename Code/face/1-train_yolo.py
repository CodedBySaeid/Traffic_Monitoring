from ultralytics import YOLO
import yaml

with open("Code/face/faces.yaml") as file:
    config = yaml.safe_load(file)


model = YOLO("yolov8s.pt")

model.train(
    data="Code/face/faces.yaml",   # dataset config
    epochs=20,            
    batch=2,             
    imgsz=640,            
    device="cpu"          # 0 for GPU, "cpu" for CPU
)

metrics = model.val()
print(metrics)

results = model.predict(
    source=config["test"],  
    conf=0.25,
    save=True,
    show=True
)

model.export(format="onnx")

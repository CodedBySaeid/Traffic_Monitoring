'''
This code shows how to use the trained model in first file to detect objects.

'''
from ultralytics import YOLO

model = YOLO("runs/detect/train10/weights/best.pt")

# Run prediction on a video
results = model.predict(
    source="detrac.mp4",  # path to video file
    conf=0.25,                 # confidence threshold
    save=True,                 # save output video with detections
    show=True                  # show video while processing (optional)
)

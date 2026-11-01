# 🚦 Vehicle Detection, Tracking & Speed Estimation System

A computer vision pipeline for traffic monitoring that detects vehicles, tracks them across frames, estimates their speed between two virtual lines, and computes traffic flow and density metrics.

The system is built on **YOLOv8** for object detection and **ByteTrack** for multi-object tracking.

![Vehicle Detection & Speed Estimation](./Rec_0040.mp4)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Inference & Speed Estimation](#inference--speed-estimation)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Output](#output)
- [Metrics Explained](#metrics-explained)
- [Notes & Limitations](#notes--limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

This project provides an end-to-end solution for traffic analysis:

1. **Train** a YOLOv8 model on a vehicle detection dataset (DETRAC).
2. **Track** vehicles using ByteTrack across video frames.
3. **Estimate speed** of each vehicle between two reference lines separated by a known real-world distance.
4. **Compute** aggregate traffic metrics:
   - Traffic flow (vehicles/hour)
   - Mean velocity (km/h)
   - Traffic density (vehicles/km)

---

## Features

- ✅ Custom-trained YOLOv8 model for vehicle detection
- ✅ ByteTrack multi-object tracking with persistent IDs
- ✅ Region of Interest (ROI) polygon masking to ignore irrelevant detections
- ✅ Per-vehicle speed estimation between two lines
- ✅ Color-coded bounding boxes (green = safe, red = overspeeding)
- ✅ Live visualization with annotation overlay
- ✅ Video output with all annotations
- ✅ Traffic flow, mean velocity, and density statistics

---

## Project Structure

```text
.
├── Code/
│   └── traffic/
│       ├── Data/
│       │   └── detrac.mp4          # Input video
│       ├── detrac.yaml             # Dataset config for training
│       ├── mybytetrack.yaml        # ByteTrack tracker config
│       └── train.py                # YOLO training script
├── runs/
│   └── detect/
│       └── train4/
│           └── weights/
│               └── best.pt         # Trained model weights
├── ByteTracker.py                  # Main tracking and speed estimation script
├── average_velocity_gpu.mp4        # Output video
└── README.md
```

---

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- PyTorch ≥ 2.0
- Ultralytics
- OpenCV
- NumPy
- PyYAML

## Installation

Install the required dependencies:

```bash
pip install ultralytics opencv-python numpy pyyaml torch torchvision
```

---

## Dataset

This project uses the **DETRAC (Detection and Tracking in Real-World Traffic Scenarios)** dataset for training.

Prepare your `detrac.yaml` file in the following format:

```yaml
path: ./Code/traffic/Data
train: images/train
val: images/val
test: images/test

names:
  0: car
  1: bus
  2: truck
  3: van
```

💡 Download DETRAC from the official site and convert it to YOLO format before training.

---

## Model Training

Run the training script:

```bash
python train.py
```

This will:

1. Load the `detrac.yaml` configuration.
2. Fine-tune a pretrained `yolov8s.pt` model for 10 epochs.
3. Validate the model on the validation set.
4. Run predictions on the test split.
5. Export the model to ONNX format.

Trained weights will be saved to:

```text
runs/detect/train/weights/best.pt
```

⚠️ **Important:** On Windows, when using `device=0` (GPU), the training code must be wrapped inside:

```python
if __name__ == "__main__":
```

This helps avoid multiprocessing errors.

---

## Inference & Speed Estimation

Run the tracking and speed estimation script:

```bash
python ByteTracker.py
```

Before running, update the paths in `ByteTracker.py`:

```python
model = YOLO("runs/detect/train4/weights/best.pt")  # Trained weights
video_path = "Code/traffic/Data/detrac.mp4"         # Input video
```

Press `q` to stop the live preview.

---

## Configuration

Key parameters in `ByteTracker.py` that you may want to adjust:

| Parameter | Description | Default |
|---|---|---|
| `line_y1` | Y-coordinate of the entry line (pixels) | `150` |
| `line_y2` | Y-coordinate of the exit line (pixels) | `300` |
| `real_distance_m` | Real-world distance between lines (meters) | `20.0` |
| `roi_polygon` | Polygon defining the monitored road region | `[(280,600),(1100,600),(570,0),(500,0)]` |
| `conf` | YOLO confidence threshold | `0.25` |
| Speed threshold | Overspeed color threshold (km/h) | `60` |

📐 **Calibration tip:** The value of `real_distance_m` should be measured on the actual road. Accurate speed estimation depends on this value being correct.

---

## How It Works

The system follows this pipeline:

```text
           ┌───────────────────────┐
           │   Video Frame Input   │
           └──────────┬────────────┘
                      │
                      ▼
           ┌───────────────────────┐
           │  YOLOv8 Detection     │
           └──────────┬────────────┘
                      │
                      ▼
           ┌───────────────────────┐
           │  ByteTrack Tracking   │
           │  (persistent IDs)     │
           └──────────┬────────────┘
                      │
                      ▼
           ┌───────────────────────┐
           │  ROI Polygon Filter   │
           └──────────┬────────────┘
                      │
                      ▼
      ┌───────────────┴────────────────┐
      │                                │
      ▼                                ▼
┌──────────────┐               ┌────────────────┐
│ Entry Line   │               │  Exit Line     │
│  crossing    │──► time ─────►│  crossing      │
└──────────────┘               └───────┬────────┘
                                       │
                                       ▼
                             ┌──────────────────┐
                             │  Speed = d / t   │
                             │  (m/s → km/h)    │
                             └────────┬─────────┘
                                      │
                                      ▼
                             ┌──────────────────┐
                             │ Traffic Metrics  │
                             │ Flow / Velocity  │
                             │    / Density     │
                             └──────────────────┘
```

### Step-by-step Process

1. Each frame is passed through a YOLOv8 detector.
2. ByteTrack assigns a persistent ID to each vehicle.
3. The vehicle's centroid is checked against the ROI mask. Out-of-ROI detections are ignored.
4. When the centroid crosses Line 1 (entry), the frame index is recorded.
5. When the centroid crosses Line 2 (exit), the elapsed time is computed.
6. Speed is calculated using the distance and elapsed time:

```python
speed_mps = real_distance_m / elapsed_seconds
speed_kmh = speed_mps * 3.6
```

7. Vehicles are drawn with color-coded bounding boxes:

   - 🟢 **Green:** Speed ≤ 60 km/h (safe)
   - 🔴 **Red:** Speed > 60 km/h (overspeeding)

---

## Output

The script produces:

### Annotated Video

`average_velocity_gpu.mp4` — annotated video containing:

- Vehicle tracking IDs
- Bounding boxes
- Speed labels
- Traffic counters

### Console Summary

Example output:

```text
Final counts: {'car': 42, 'truck': 8, 'bus': 3}
Traffic flow: 3180.0 veh/hr
Velocity mean: 54.32 km/hr
Traffic density: 58.54 veh/km
```

---

## Metrics Explained

| Metric | Formula | Description |
|---|---|---|
| Traffic Flow (q) | `total_vehicles × 60` | Vehicles counted over the analysis window, scaled to an hourly rate |
| Mean Velocity (v̄) | `mean(speeds)` | Average speed of all tracked vehicles |
| Traffic Density (k) | `q / v̄` | Vehicles per kilometer of roadway |

**Note:** The scaling factor `× 60` assumes the video segment spans approximately one minute. Adjust it according to the duration of your footage.

---

## Notes & Limitations

- 🎯 **Speed accuracy:** Depends heavily on correct calibration of `real_distance_m` and a top-down or near-orthogonal camera angle.
- 🚗 **Line-crossing detection:** Uses a ±5 pixel tolerance. Vehicles that move very fast between frames may be missed. Consider interpolating the exact crossing point for higher accuracy.
- 🌍 **Perspective distortion:** Is not corrected. For production use, apply a homography transformation to obtain a bird's-eye view.
- 🚦 **ROI configuration:** The ROI polygon must be redefined for each camera setup.
- ⚙️ **Tracking configuration:** The current ByteTrack configuration is loaded from `Code/traffic/mybytetrack.yaml`. Tune `track_buffer`, `match_thresh`, and `track_high_thresh` for your scene.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [DETRAC Dataset](https://detrac-db.rit.albany.edu/)

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.

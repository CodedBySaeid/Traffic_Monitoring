'''
This code uses built it bytetracker to track objects. then counts number of passing vehicles.

'''

def SortTracker():
    """
    from ultralytics import YOLO
    import cv2
    from typing import Set
    import numpy as np
    from sort import Sort # pip install sort-tracker (needs visual studio)

    # Load model
    model = YOLO("runs/detect/train/weights/best.pt")

    # Load tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Open video
    cap = cv2.VideoCapture("input_video.mp4")

    # Output writer
    out = cv2.VideoWriter("output_counted.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        cap.get(cv2.CAP_PROP_FPS),
                        (int(cap.get(3)), int(cap.get(4))))

    # Counting setup
    count_line_y = 400   # Y-coordinate for counting line
    vehicle_count = 0
    memory = set()       # store counted IDs

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, conf=0.3, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck (COCO IDs)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append([x1, y1, x2, y2, conf])

        # Update tracker
        detections = np.array(detections)
        tracked = tracker.update(detections)

        # Draw line
        cv2.line(frame, (0, count_line_y), (frame.shape[1], count_line_y), (0, 255, 255), 2)

        # Process tracked objects
        for x1, y1, x2, y2, obj_id in tracked:
            x1, y1, x2, y2, obj_id = map(int, [x1, y1, x2, y2, obj_id])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Check if vehicle crosses line
            if cy > count_line_y - 5 and cy < count_line_y + 5:
                if obj_id not in memory:
                    vehicle_count += 1
                    memory.add(obj_id)

        # Display count
        cv2.putText(frame, f"Count: {vehicle_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(frame)
        cv2.imshow("Vehicle Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Total vehicles counted: {vehicle_count}")
    """




def ByteTracker():
    import cv2
    from ultralytics import YOLO
    from collections import defaultdict

    # Load YOLOv8 model (replace with your trained weights if needed)
    model = YOLO("runs/detect/train10/weights/best.pt") # or "best.pt" if you trained your own

    # Class names (COCO dataset default, replace if you trained custom classes)
    class_names = model.model.names

    # Video path
    video_path = "Code/traffic/Data/detrac.mp4"
    cap = cv2.VideoCapture(video_path)

    # Output video (optional)
    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(3)), int(cap.get(4))),
    )

    # Line position (y-coordinate)
    count_line_y = 400  # adjust to your scene

    # Vehicle counts per class
    vehicle_counts = defaultdict(int)

    # Track previous positions
    last_positions = {}

    # Memory of counted IDs
    memory = defaultdict(set)

    # Recent crossing memory for duplicate suppression
    recent_counts = []

    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Run YOLO tracking with ByteTrack
        results = model.track(
            frame,
            persist=True,
            conf=0.25,
            tracker="Code/traffic/mybytetrack.yaml"
        )

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                class_name = class_names[cls]

                # Draw bounding box + ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (250, 0, 50), 2)
                cv2.putText(frame, f"{class_name} {track_id}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (250, 0, 50), 2)

                # --- Line crossing detection ---
                if track_id in last_positions:
                    prev_cy = last_positions[track_id]

                    if prev_cy < count_line_y and cy >= count_line_y:
                        # --- Duplicate suppression ---
                        duplicate = False
                        for (px, py, pclass, pf) in recent_counts:
                            if (pclass == class_name
                                and abs(cx - px) < 20   # horizontal tolerance
                                and abs(cy - py) < 20   # vertical tolerance
                                and frame_num - pf < 15):  # time window
                                duplicate = True
                                break

                        if not duplicate:
                            vehicle_counts[class_name] += 1
                            memory[class_name].add(track_id)
                            recent_counts.append((cx, cy, class_name, frame_num))

                last_positions[track_id] = cy

        # Draw counting line
        cv2.line(frame, (350, count_line_y), (925, count_line_y),
                (0, 0, 255), 2)

        # Show counts
        y_offset = 30
        for cls, count in vehicle_counts.items():
            cv2.putText(frame, f"{cls}: {count}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            y_offset += 30

        out.write(frame)
        cv2.imshow("Vehicle Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Final counts:", dict(vehicle_counts))


ByteTracker()
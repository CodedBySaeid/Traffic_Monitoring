def ByteTracker():
    import cv2
    from ultralytics import YOLO
    from collections import defaultdict
    import numpy as np 
    # Load YOLOv8 model
    model = YOLO("runs/detect/train4/weights/best.pt")
    class_names = model.model.names

    # Video path
    video_path = "detrac.mp4"
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video
    out = cv2.VideoWriter(
        "average_velocity.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (int(cap.get(3)), int(cap.get(4))),
    )

    # Two horizontal lines (adjust as needed)
    line_y1 = 150  # entry line
    line_y2 = 300  # exit line
    real_distance_m = 20.0  # real-world distance between the lines

    safe_color = (0,255,0)
    unsafe_color = (0,0,255)
    
    # Define polygon points of the road region
    roi_polygon = [(400, 300), (800, 300), (450, 150), (700, 150)]


    # Vehicle counts
    vehicle_counts = defaultdict(int)

    # Track line-crossing times
    entry_times = {}   # {id: frame_num when crossed first line}
    speeds = {}        # {id: speed_kmh}

    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # # Create mask
        # mask = cv2.fillPoly(
        #     np.zeros(frame.shape[:2], dtype=np.uint8), 
        #     [np.array(roi_polygon, np.int32)], 
        #     1
        # )
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
                # if mask[cy, cx] == 0:
                #     continue  # skip detections outside highway
                class_name = class_names[cls]

                # --- Line 1 (entry) ---
                if cy >= line_y1 - 5 and cy <= line_y1 + 5:
                    if track_id not in entry_times:  # log only first crossing
                        entry_times[track_id] = frame_num

                # --- Line 2 (exit) ---
                if cy >= line_y2 - 5 and cy <= line_y2 + 5:
                    if track_id in entry_times and track_id not in speeds:
                        time_frames = frame_num - entry_times[track_id]
                        time_s = time_frames / fps
                        if time_s > 0:
                            speed_mps = real_distance_m / time_s
                            speed_kmh = speed_mps * 3.6
                            speeds[track_id] = speed_kmh
                            vehicle_counts[class_name] += 1


                color = safe_color
                # --- Draw bounding box ---
                label = f"{class_name} {track_id}"
                if track_id in speeds:
                    label += f" {speeds[track_id]:.1f} km/h"
                    if speeds[track_id] > 60:
                        color = unsafe_color
                        
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw counting lines
        cv2.line(frame, (450, line_y1), (700, line_y1), (255, 255, 255), 2)
        cv2.line(frame, (400, line_y2), (800, line_y2), (255, 255, 255), 2)

        # Show counts
        y_offset = 30
        for cls, count in vehicle_counts.items():
            cv2.putText(frame, f"{cls}: {count}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            y_offset += 30

        out.write(frame)
        cv2.imshow("Vehicle Counter + Speed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Final counts:", dict(vehicle_counts))




ByteTracker()
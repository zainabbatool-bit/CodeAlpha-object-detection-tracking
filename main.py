import time
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------- Settings ----------
VIDEO_SOURCE = 0            # 0 = webcam, ya video file: "video.mp4"
CONF_THRESHOLD = 0.4        # Is se kam confidence wali detection ignore hogi
SAVE_OUTPUT = True          # True = result video save hogi
OUTPUT_FILE = "output.mp4"  # Save hone wali file ka naam
# ------------------------------

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)


def get_color(track_id):
    """Har ID ke liye alag color banata hai."""
    n = int(track_id) if str(track_id).isdigit() else 0
    return ((n * 37) % 255, (n * 91) % 255, (n * 53) % 255)


cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: Webcam ya video file nahi khuli.")
    exit()

# Video ka size pata karo (save karne ke liye zaroori hai)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 20, (frame_w, frame_h))

print("Shuru ho gaya. Band karne ke liye 'q' dabao.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video khatam ya frame nahi mila.")
        break

    # 1) YOLO se detection
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = model.names[int(box.cls[0])]
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

    # 2) Deep SORT se tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # 3) Boxes, labels aur ID draw karo
    count = 0
    for track in tracks:
        if not track.is_confirmed():
            continue

        count += 1
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        class_name = track.get_det_class() or "object"
        label = f"{class_name} ID:{track_id}"
        color = get_color(track_id)

        cv2.rectangle(frame, (l, t), (r, b), color, 2)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (l, t - h - 10), (l + w, t), color, -1)
        cv2.putText(frame, label, (l, t - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 4) FPS aur objects ki ginti
    now = time.time()
    fps = 1 / (now - prev_time) if now > prev_time else 0
    prev_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 5) Save aur display
    if writer is not None:
        writer.write(frame)

    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if writer is not None:
    writer.release()
    print(f"Video save ho gayi: {OUTPUT_FILE}")
cv2.destroyAllWindows()
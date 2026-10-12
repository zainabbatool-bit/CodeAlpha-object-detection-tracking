import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------- Settings ----------
VIDEO_SOURCE = 0          # 0 = webcam, ya video file: "video.mp4"
CONF_THRESHOLD = 0.4      # Is se kam confidence wali detection ignore hogi
# ------------------------------

# YOLO model (detection ke liye)
model = YOLO("yolov8n.pt")

# Deep SORT tracker (max_age = kitne frames tak gayab object ko yaad rakhe)
tracker = DeepSort(max_age=30)


def get_color(track_id):
    """Har ID ke liye alag color banata hai."""
    n = int(track_id) if str(track_id).isdigit() else 0
    return ((n * 37) % 255, (n * 91) % 255, (n * 53) % 255)


cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: Webcam ya video file nahi khuli.")
    exit()

print("Tracking shuru. Band karne ke liye 'q' dabao.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame nahi mila, band kar rahe hain.")
        break

    # Step A: YOLO se detection
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = model.names[int(box.cls[0])]

        # Deep SORT ko format chahiye: ([x, y, width, height], confidence, class)
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

    # Step B: Deep SORT se tracking (har object ko ID milti hai)
    tracks = tracker.update_tracks(detections, frame=frame)

    # Step C: Boxes aur ID draw karo
    for track in tracks:
        if not track.is_confirmed():
            continue

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

    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
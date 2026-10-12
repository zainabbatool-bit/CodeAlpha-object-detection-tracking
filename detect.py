import cv2
from ultralytics import YOLO

# ---------- Settings ----------
VIDEO_SOURCE = 0          # 0 = webcam, ya video file: "video.mp4"
CONF_THRESHOLD = 0.4      # Is se kam confidence wali detection ignore hogi
# ------------------------------

# Pre-trained YOLOv8 nano model (pehli baar apne aap download hoga)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: Webcam ya video file nahi khuli.")
    exit()

print("Detection shuru. Band karne ke liye 'q' dabao.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame nahi mila, band kar rahe hain.")
        break

    # Frame par YOLO chalao
    results = model(frame, verbose=False)[0]

    # Har detected object ke liye box aur label draw karo
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = f"{model.names[class_id]} {conf:.2f}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label ke peeche green background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)

        # Label text (black)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
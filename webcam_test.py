import cv2

# 0 = laptop ka default webcam
# Video file use karni ho to: VIDEO_SOURCE = "video.mp4"
VIDEO_SOURCE = 0

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: Webcam ya video file nahi khuli.")
    exit()

print("Webcam chal gaya. Band karne ke liye 'q' dabao.")

while True:
    ret, frame = cap.read()

    # Agar frame nahi mila (video khatam ya webcam issue)
    if not ret:
        print("Frame nahi mila, band kar rahe hain.")
        break

    cv2.imshow("Webcam Test", frame)

    # 'q' dabane par loop band
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
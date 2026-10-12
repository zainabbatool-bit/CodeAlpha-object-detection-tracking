# Object Detection and Tracking

Real-time object detection and tracking using **YOLOv8** and **Deep SORT**. The program reads video from a webcam or a video file, detects objects in every frame, and gives each object a unique tracking ID that stays the same as the object moves.

## Features

- Real-time video input from webcam or video file (OpenCV)
- Object detection with pre-trained YOLOv8 (80 everyday object classes)
- Bounding boxes with class name and confidence
- Object tracking with Deep SORT (a unique ID for each object)
- Live display with labels, tracking IDs, FPS and object count
- Optional saving of the output video

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- Deep SORT (deep-sort-realtime)
- PyTorch

## How It Works

1. OpenCV reads each frame from the webcam or video file.
2. YOLOv8 detects objects in the frame and returns bounding boxes.
3. The detections are passed to Deep SORT, which matches them with objects from earlier frames and assigns tracking IDs.
4. The frame is displayed with boxes, labels and IDs drawn on it.

## Installation

1. Clone this repository:

```
git clone https://github.com/zainabbatool-bit/object-detection-tracking.git
cd object-detection-tracking
```

2. Create and activate a virtual environment (Windows):

```
python -m venv venv
venv\Scripts\activate
```

3. Install the requirements:

```
pip install -r requirements.txt
```

## Usage

Run the final version:

```
python main.py
```

Press **q** in the video window to quit.

To use a video file instead of the webcam, open `main.py` and change this line:

```python
VIDEO_SOURCE = "video.mp4"   # default is 0 (webcam)
```

Other settings at the top of `main.py`:

- `CONF_THRESHOLD`: minimum confidence for a detection (default 0.4)
- `SAVE_OUTPUT`: set to `True` to save the result as `output.mp4`

> The YOLOv8 model file (`yolov8n.pt`) downloads automatically the first time you run the program, so an internet connection is needed once.

## Project Structure

```
object-detection-tracking/
├── main.py             # Final version: detection + tracking + FPS + saving
├── track.py            # Detection + tracking
├── detect.py           # Detection only
├── webcam_test.py      # Webcam test
├── requirements.txt    # Libraries needed
└── README.md
```

## Author

**Zainab Batool**
GitHub: [zainabbatool-bit](https://github.com/zainabbatool-bit)
- LinkedIn: [Zainab Batool](https://www.linkedin.com/in/zainab-batool-b4661734a)
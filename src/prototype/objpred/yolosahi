
!pip install torch torchvision
!pip install yolov5
!pip install sahi
!pip install opencv-python
!pip install opencv-python-headless 

#to clone,change directory and install the ereqquirements from yolov5 repositry
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -U -r requirements.txt

import cv2
import torch
from sahi.predict import Predictor
from sahi import AutoDetectionModel
from sahi.utils.file import check_and_create_directory

model = torch.hub.load('ultralytics/yolov5:v5.0', 'yolov5s')  # Change this for other YOLOv5 models (e.g., yolov5m, yolov5l)(s has less no of parameters to get accurate and precision in real time).
detection_model = AutoDetectionModel.from_pretrained("yolov5s", model_type="yolov5")  # Use the correct YOLOv5 model
predictor = Predictor(detection_model) 

cap = cv2.VideoCapture(0)  # Use `0` for the default webcam or provide video file path

while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # Exit if video stream ends
    
    sahi_result = predictor.predict(frame)

    for pred in sahi_result.predictions:
        box = pred.box
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        label = f"{pred.category_name} {pred.score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Show the frame with bounding boxes and labels
    cv2.imshow('Real-time YOLOv5 Prediction with SAHI', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

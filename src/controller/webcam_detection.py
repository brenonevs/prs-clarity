from ultralytics import YOLO
import cv2
import logging
import threading
import time

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

class WebcamDetection:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.model = YOLO(r"src\models\yolo11n.pt")
        self.real_time_objects = []
        self.start_time = time.time()
        self.start_reset_timer()

    def start_reset_timer(self):
        def reset_list():
            while True:
                threading.Event().wait(10)
                self.real_time_objects = []

        threading.Thread(target=reset_list, daemon=True).start()

    def detect(self):

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame, stream=True)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    label = f"{self.model.names[cls]} {conf:.2f}"
                    self.real_time_objects.append(label)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            current_time = time.time()
            if current_time - self.start_time >= 10:
                print(f'Real time objects: {self.real_time_objects}')
                self.start_time = time.time()

            cv2.imshow("YOLOv11 - Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

detection = WebcamDetection()
detection.detect()

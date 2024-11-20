from ultralytics import YOLO
from depth_estimation import DepthEstimator
import torch
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)  # Configura o dispositivo do modelo
        print(f"Using device: {device}")

        self.depth_estimator = DepthEstimator(model_type="DPT_Hybrid", use_manual_limits=True, depth_min=5, depth_max=20)

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

            # Processar detecções
            results = self.model(frame, stream=True)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box
                    conf = box.conf[0]  # Confiança da detecção
                    cls = int(box.cls[0])  # Classe detectada
                    label = self.model.names[cls]  # Nome do objeto detectado
                    label_with_conf = f"{label} {conf:.2f}"

                    # Adicionar o objeto com a profundidade à lista
                    distance_info = f"{label_with_conf}"
                    self.real_time_objects.append(distance_info)

                    # Desenhar a bounding box e a distância no quadro
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_with_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Exibir a lista de objetos detectados em tempo real a cada 10 segundos
            current_time = time.time()
            if current_time - self.start_time >= 10:
                print(f"Objetos em tempo real: {self.real_time_objects}")
                self.real_time_objects = []  # Reiniciar a lista de objetos
                self.start_time = current_time

            # Mostrar a saída
            cv2.imshow("YOLOv11 - Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

detection = WebcamDetection()
detection.detect()

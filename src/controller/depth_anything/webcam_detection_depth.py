import os
import sys
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import threading
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

CONFIDENCE_THRESHOLD = 0.6  # Limite mínimo de confiança para considerar detecções

class WebcamDetectionWithDepth:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.model = YOLO(r"../models/yolo11n.pt")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        self.depth_model = self._load_depth_model()
        self.transform = self._get_transform()

        self.real_time_objects = []
        self.previous_detections = {}
        self.detection_history = {}
        self.smoothed_bboxes = {}
        self.start_time = time.time()
        self.start_reset_timer()

    def _load_depth_model(self):
        local_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "torchhub/facebookresearch_dinov2_main"))
        print(f"Loading model from: {local_model_path}")
        depth_model = torch.hub.load(local_model_path, 'dinov2_vits14', source='local').to(self.device)
        depth_model.eval()
        return depth_model

    @staticmethod
    def _get_transform():
        return Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def _process_depth(self, frame, bbox):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        image = self.transform({'image': frame_rgb})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.depth_model(image)

        if depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.ndim == 3:
            depth = depth.unsqueeze(0)

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        depth_resized = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)
        bbox_depth = depth_resized[0, 0, y1:y2, x1:x2]

        if bbox_depth.numel() == 0:
            return "Unknown"

        avg_depth = bbox_depth.mean().item()
        threshold = 0.4
        return "Near" if avg_depth < threshold else "Far"

    def _smooth_bbox(self, label, new_bbox):
        if label not in self.smoothed_bboxes:
            self.smoothed_bboxes[label] = new_bbox
            return new_bbox

        old_bbox = self.smoothed_bboxes[label]
        smoothed_bbox = [
            int(0.7 * old_bbox[i] + 0.3 * new_bbox[i])
            for i in range(4)
        ]
        self.smoothed_bboxes[label] = smoothed_bbox
        return smoothed_bbox

    def _update_detection_history(self, label, conf):
        if label not in self.detection_history:
            self.detection_history[label] = []
        self.detection_history[label].append(conf)
        if len(self.detection_history[label]) > 10:  # Limitar a 10 frames
            self.detection_history[label].pop(0)

        # Converter tensores para valores na CPU antes de calcular a média
        conf_values = [conf.cpu().item() if torch.is_tensor(conf) else conf for conf in self.detection_history[label]]
        return np.mean(conf_values)

    def start_reset_timer(self):
        def reset_list():
            while True:
                threading.Event().wait(10)
                self.real_time_objects = []

        threading.Thread(target=reset_list, daemon=True).start()

    def detect_with_depth(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame, stream=True)
            current_detections = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    label = self.model.names[cls]

                    if conf < CONFIDENCE_THRESHOLD:
                        continue

                    avg_conf = self._update_detection_history(label, conf)
                    if avg_conf < CONFIDENCE_THRESHOLD:
                        continue

                    smoothed_bbox = self._smooth_bbox(label, (x1, y1, x2, y2))
                    proximity = self._process_depth(frame, smoothed_bbox)

                    current_detections.append((label, smoothed_bbox))
                    self.real_time_objects.append(f"{label} {avg_conf:.2f}, {proximity}")

                    cv2.rectangle(frame, (smoothed_bbox[0], smoothed_bbox[1]), (smoothed_bbox[2], smoothed_bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}: {proximity}", (smoothed_bbox[0], smoothed_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.previous_detections = {label: bbox for label, bbox in current_detections}

            cv2.imshow("Object Detection with Proximity", frame)

            current_time = time.time()
            if current_time - self.start_time >= 10:
                print(f"Real-time objects: {self.real_time_objects}")
                self.real_time_objects = []
                self.start_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detection_with_depth = WebcamDetectionWithDepth()
    detection_with_depth.detect_with_depth()

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


class DepthDetectionApp:
    def __init__(self, encoder='vits', video_path=0):
        self.encoder = encoder
        self.video_path = video_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.depth_model = self._load_model()
        self.transform = self._get_transform()

    def _load_model(self):
        depth_model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{self.encoder}14').to(self.device)
        depth_model.eval()
        total_params = sum(param.numel() for param in depth_model.parameters())
        print(f'Total parameters: {total_params / 1e6:.2f}M')
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

    def _process_frame(self, frame):
        """Process a single frame to compute depth."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        h, w = frame.shape[:2]
        image = self.transform({'image': frame})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.depth_model(image)

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        return depth_color

    def run(self):
        """Run the depth detection application."""
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("Error: Unable to open video.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            depth_color = self._process_frame(frame)

            # Display the depth map
            cv2.imshow('Depth Detection', depth_color)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = DepthDetectionApp(encoder='vits', video_path=0)
    app.run()

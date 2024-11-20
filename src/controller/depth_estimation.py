import os
import torch
import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

class DepthEstimator:
    def __init__(self, model_type="DPT_Hybrid", use_manual_limits=True, depth_min=5, depth_max=20):
        self.model_type = model_type
        self.use_manual_limits = use_manual_limits
        self.depth_min = depth_min
        self.depth_max = depth_max

        # Verificar se GPU está disponível
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

        # Carregar o modelo MiDaS
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device)
        self.model.eval()

        # Configurar transformações
        self.transform = self._get_transform()

    def _get_transform(self):
        """Define as transformações baseadas no modelo."""
        if "DPT" in self.model_type:
            return Compose([
                Resize((384, 384)),
                ToTensor(),
                Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            return Compose([
                Resize((384, 384)),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def load_image(self, image_path):
        """Carrega e prepara a imagem."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        # Carregar a imagem com OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Falha ao carregar a imagem: {image_path}")

        # Converter para RGB e depois para PIL Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def estimate_depth(self, image_path):
        """Estima a profundidade da imagem."""
        img = self.load_image(image_path)
        input_batch = self.transform(img).unsqueeze(0).to(self.device)  # Enviar para GPU

        # Estimar profundidade
        with torch.no_grad():
            depth = self.model(input_batch).squeeze().cpu().numpy()  # Retornar para CPU
        return depth
    
    def estimate_depth_from_frame(self, image):
        """Estima a profundidade diretamente de uma matriz de imagem."""
        # Converter a matriz OpenCV (BGR) para RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Converter a imagem para um formato compatível com o modelo (PIL)
        img_pil = Image.fromarray(img_rgb)

        # Aplicar transformações
        input_batch = self.transform(img_pil).unsqueeze(0).to(self.device)

        # Estimar profundidade
        with torch.no_grad():
            depth = self.model(input_batch).squeeze().cpu().numpy()

        return depth

    def normalize_depth(self, depth):
        """Normaliza o mapa de profundidade."""
        if self.use_manual_limits:
            # Normalização com limites manuais
            depth_clamped = depth.clip(min=self.depth_min, max=self.depth_max)
            depth_normalized = (depth_clamped - self.depth_min) / (self.depth_max - self.depth_min) * 255
        else:
            # Normalização automática
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        
        return depth_normalized.clip(0, 255)
    
    def visualize_depth(self, depth_normalized):
        """Aplica um mapa de cores e exibe a profundidade."""
        depth_colormap = cv2.applyColorMap(depth_normalized.astype('uint8'), cv2.COLORMAP_JET)
        cv2.imshow("Depth Estimation", depth_colormap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def print_depth_info(self, depth):
        """Exibe informações detalhadas sobre o mapa de profundidade."""
        print(f"Profundidade mínima (original): {depth.min()}")
        print(f"Profundidade máxima (original): {depth.max()}")
        if self.use_manual_limits:
            print(f"Limites aplicados - Mínimo: {self.depth_min}, Máximo: {self.depth_max}")

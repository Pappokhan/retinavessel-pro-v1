import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
import logging
import json
from pathlib import Path
from datetime import datetime


# Custom transforms to avoid torchvision import issues
class CustomTransforms:
    """Custom image transforms to replace torchvision"""

    @staticmethod
    def resize(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Resize image"""
        return image.resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to torch.Tensor"""
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)

        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=2)

        # Convert HWC to CHW
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        # Normalize to [0, 1]
        tensor = tensor / 255.0

        return tensor

    @staticmethod
    def normalize(tensor: torch.Tensor,
                  mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                  std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> torch.Tensor:
        """Normalize tensor with mean and std"""
        mean_tensor = torch.tensor(mean).view(3, 1, 1)
        std_tensor = torch.tensor(std).view(3, 1, 1)
        return (tensor - mean_tensor) / std_tensor

    @classmethod
    def compose(cls, transforms_list):
        """Compose multiple transforms"""

        def composed(image):
            for transform in transforms_list:
                image = transform(image)
            return image

        return composed


# Import model after defining transforms
from .unet import AttentionUNet

logger = logging.getLogger(__name__)


class ModelInference:
    """Model inference service for retinal vessel segmentation"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model = None
        self.transform = None
        self.metadata = {}

        self._load_model()
        self._setup_transforms()

    def _load_model(self):
        """Load the trained model"""
        try:
            logger.info(f"Loading model from {self.config.MODEL_PATH}")

            # Check if model file exists
            model_path = Path(self.config.MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Initialize model architecture
            self.model = AttentionUNet(
                n_channels=3,
                n_classes=1,
                bilinear=True,
                dropout=0.1
            )

            # Load model weights
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            except Exception as e:
                logger.warning(f"Could not load with weights_only=False: {e}")
                state_dict = torch.load(model_path, map_location=self.device)

            # Handle different state dict formats
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    # Handle nested state dict
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']

                # Remove 'module.' prefix if present (for DataParallel models)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                # Load state dict
                self.model.load_state_dict(state_dict)
            else:
                logger.warning("Unexpected model format, using random initialization")

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")

            # Try to load metadata
            metadata_path = model_path.parent / "model_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _setup_transforms(self):
        """Setup image transformations for inference"""
        self.transform = CustomTransforms.compose([
            lambda img: CustomTransforms.resize(img, self.config.IMG_SIZE),
            CustomTransforms.to_tensor,
            lambda tensor: CustomTransforms.normalize(tensor)
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image, threshold: float = 0.5) -> Tuple[np.ndarray, float]:
        """
        Run inference on an image

        Args:
            image: PIL Image
            threshold: Probability threshold for binary mask

        Returns:
            Tuple of (binary_mask, confidence_score)
        """
        start_time = datetime.now()

        try:
            # Preprocess image
            tensor = self.preprocess(image)

            # Run inference
            with torch.no_grad():
                logits = self.model(tensor)
                probabilities = torch.sigmoid(logits)

                # Calculate confidence as mean probability
                confidence = float(torch.mean(probabilities).cpu().numpy())

                # Convert to numpy
                prob_map = probabilities.squeeze().cpu().numpy()

            # Create binary mask
            binary_mask = (prob_map > threshold).astype(np.uint8)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Inference complete - Confidence: {confidence:.3f}, Time: {processing_time:.2f}s")

            return binary_mask, confidence

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise

    def batch_predict(self, images: list, threshold: float = 0.5) -> list:
        """Run inference on multiple images"""
        results = []
        for img in images:
            try:
                mask, confidence = self.predict(img, threshold)
                results.append({
                    "mask": mask,
                    "confidence": confidence,
                    "success": True
                })
            except Exception as e:
                logger.error(f"Batch prediction failed for image: {e}")
                results.append({
                    "mask": None,
                    "confidence": 0.0,
                    "success": False,
                    "error": str(e)
                })
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "architecture": "Attention U-Net",
            "input_size": self.config.IMG_SIZE,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "threshold": self.config.THRESHOLD,
            "loaded_at": datetime.now().isoformat()
        }

        # Add metadata if available
        if self.metadata:
            info["metadata"] = self.metadata

        return info

    def validate_model(self) -> bool:
        """Validate model functionality"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, *self.config.IMG_SIZE).to(self.device)

            # Test forward pass
            with torch.no_grad():
                output = self.model(dummy_input)

            # Check output shape
            expected_shape = (1, 1, *self.config.IMG_SIZE)
            if output.shape != expected_shape:
                logger.error(f"Unexpected output shape: {output.shape}, expected: {expected_shape}")
                return False

            logger.info("Model validation successful")
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
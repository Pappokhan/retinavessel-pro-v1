import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image processing utilities"""

    def __init__(self, config):
        self.config = config

    def load_image(self, file_object) -> Image.Image:
        """Load image from file object"""
        try:
            image = Image.open(file_object).convert("RGB")
            logger.info(f"Loaded image: {image.size}, mode: {image.mode}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def resize(self, image: Image.Image, size: Tuple[int, int] = None) -> Image.Image:
        """Resize image"""
        if size is None:
            size = self.config.IMG_SIZE
        return image.resize(size, Image.Resampling.LANCZOS)

    def enhance(self, image: Image.Image) -> Image.Image:
        """Enhance image quality"""
        # Adjust contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)

        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)

        return image

    def assess_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Assess image quality"""
        img_array = np.array(image)

        # Calculate metrics
        brightness = np.mean(img_array) / 255.0
        contrast = np.std(img_array) / 255.0

        # Sharpness (gradient magnitude)
        from scipy.ndimage import sobel
        gray = np.mean(img_array, axis=2)
        gradient_x = sobel(gray, axis=0)
        gradient_y = sobel(gray, axis=1)
        sharpness = np.mean(np.sqrt(gradient_x ** 2 + gradient_y ** 2))

        # Determine quality
        quality_score = (
                min(brightness * 2, 1.0) * 0.4 +
                min(contrast * 3, 1.0) * 0.4 +
                min(sharpness / 100, 1.0) * 0.2
        )

        if quality_score > 0.7:
            quality = "Excellent"
        elif quality_score > 0.5:
            quality = "Good"
        elif quality_score > 0.3:
            quality = "Fair"
        else:
            quality = "Poor"

        return {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "sharpness": float(sharpness),
            "quality_score": float(quality_score),
            "quality": quality
        }


def create_overlay(original: Image.Image, mask: np.ndarray,
                   color: Tuple[int, int, int] = (220, 60, 60),
                   alpha: float = 0.6) -> Image.Image:
    """Create overlay of mask on image"""
    # Ensure mask is 2D
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Resize mask if needed
    if mask.shape != original.size[::-1]:
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(original.size, Image.Resampling.NEAREST)
        mask = np.array(mask_img)

    # Convert to arrays
    original_array = np.array(original)

    # Create overlay
    overlay = original_array.copy()

    # Apply color with transparency
    mask_indices = mask > 0
    for c in range(3):
        overlay[mask_indices, c] = (
                (1 - alpha) * overlay[mask_indices, c] +
                alpha * color[c]
        ).astype(np.uint8)

    return Image.fromarray(overlay)
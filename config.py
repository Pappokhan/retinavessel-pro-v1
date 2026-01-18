import os
from pathlib import Path
from typing import List, Tuple
import torch


class Config:
    """Configuration class for RetinaVessel Pro"""

    def __init__(self):
        # Base paths
        self.BASE_DIR = Path(__file__).parent
        self.MODEL_PATH = str(self.BASE_DIR / "models" / "attention_unet.pth")
        self.DATA_DIR = self.BASE_DIR / "data"
        self.STATIC_DIR = self.BASE_DIR / "static"

        # Model settings
        self.IMG_SIZE = (256, 256)
        self.THRESHOLD = 0.5
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Feature extraction
        self.MIN_VESSEL_AREA = 50
        self.THIN_THRESHOLD = 2.0
        self.THICK_THRESHOLD = 5.0

        # Clinical thresholds
        self.DENSITY_LOW = 0.05
        self.DENSITY_HIGH = 0.15
        self.TORTUOSITY_HIGH = 1.8
        self.BRANCHING_LOW = 100
        self.BRANCHING_HIGH = 350

        # Application settings
        self.ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "tiff", "tif"]
        self.MAX_UPLOAD_SIZE_MB = 50

        # Visualization
        self.DEFAULT_COLOR = (220, 60, 60)
        self.OVERLAY_ALPHA = 0.6

        # MLops settings
        self.ENABLE_MLOPS = True
        self.LOG_LEVEL = "INFO"
        self.MONITORING_ENABLED = True

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.BASE_DIR / "models",
            self.DATA_DIR / "raw",
            self.DATA_DIR / "processed",
            self.DATA_DIR / "outputs",
            self.STATIC_DIR / "css",
            self.STATIC_DIR / "images",
            self.BASE_DIR / "logs"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def is_gpu(self):
        """Check if GPU is available"""
        return self.DEVICE == "cuda"

    @property
    def model_exists(self):
        """Check if model file exists"""
        return Path(self.MODEL_PATH).exists()

    def get_log_file(self):
        """Get log file path"""
        return self.BASE_DIR / "logs" / "app.log"

    def get_output_dir(self):
        """Get output directory"""
        return self.DATA_DIR / "outputs"
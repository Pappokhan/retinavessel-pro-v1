import re
from typing import Tuple, List
import numpy as np
from PIL import Image


def validate_image(file_object, allowed_extensions: List[str], max_size_mb: int) -> Tuple[bool, str]:
    """Validate uploaded image"""
    # Check extension
    filename = file_object.name.lower()
    if not any(filename.endswith(f".{ext}") for ext in allowed_extensions):
        return False, f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"

    # Check size
    file_size = len(file_object.getvalue()) / (1024 * 1024)
    if file_size > max_size_mb:
        return False, f"File too large. Maximum: {max_size_mb}MB"

    # Check image
    try:
        file_object.seek(0)
        img = Image.open(file_object)
        img.verify()
        file_object.seek(0)

        # Check dimensions
        if img.width < 100 or img.height < 100:
            return False, "Image too small. Minimum: 100×100 pixels"

        if img.width > 10000 or img.height > 10000:
            return False, "Image too large. Maximum: 10000×10000 pixels"

        return True, "Image validated"

    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def validate_threshold(threshold: float) -> bool:
    """Validate threshold value"""
    return 0.0 <= threshold <= 1.0


def sanitize_filename(filename: str) -> str:
    """Sanitize filename"""
    # Remove unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Limit length
    if len(filename) > 255:
        import os
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext

    return filename
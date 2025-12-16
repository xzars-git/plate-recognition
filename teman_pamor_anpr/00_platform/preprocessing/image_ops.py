"""
Shared Preprocessing: Image Operations

Common image preprocessing used across all models:
- Normalization
- Resizing
- Color space conversion
"""

import cv2
import numpy as np
from typing import Tuple


def normalize_image(image: np.ndarray, method: str = "standard") -> np.ndarray:
    """
    Normalize image pixel values
    
    Args:
        image: Input image (0-255)
        method: "standard" (0-1) or "imagenet" (ImageNet stats)
    
    Returns:
        Normalized image
    """
    if method == "standard":
        # Simple 0-1 normalization
        return image.astype(np.float32) / 255.0
    
    elif method == "imagenet":
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        normalized = image.astype(np.float32) / 255.0
        normalized = (normalized - mean) / std
        
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int],
    padding: bool = True
) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: (width, height)
        padding: If True, add padding to match target size
    
    Returns:
        Resized image
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if padding:
        # Add padding to match target size
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        return padded
    
    return resized


def convert_color_space(image: np.ndarray, target: str) -> np.ndarray:
    """Convert between color spaces"""
    
    if target.upper() == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif target.upper() == "GRAY":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif target.upper() == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError(f"Unknown color space: {target}")

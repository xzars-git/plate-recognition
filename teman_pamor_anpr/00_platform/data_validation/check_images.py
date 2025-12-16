"""
Data Validation: Image Quality Checks

Validates image quality before training:
- Resolution check
- File integrity
- Format validation
- Duplicate detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib


class ImageValidator:
    """Validate image quality and integrity"""
    
    def __init__(self, min_width: int = 50, min_height: int = 50):
        self.min_width = min_width
        self.min_height = min_height
        self.seen_hashes = set()
    
    def validate_image(self, image_path: Path) -> Dict[str, any]:
        """
        Validate single image
        
        Returns:
            dict: Validation results
        """
        result = {
            "path": str(image_path),
            "valid": True,
            "errors": []
        }
        
        # Check file exists
        if not image_path.exists():
            result["valid"] = False
            result["errors"].append("File not found")
            return result
        
        # Check file size
        file_size = image_path.stat().st_size
        if file_size == 0:
            result["valid"] = False
            result["errors"].append("Empty file")
            return result
        
        # Try loading image
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                result["valid"] = False
                result["errors"].append("Cannot read image")
                return result
            
            # Check dimensions
            height, width = img.shape[:2]
            if width < self.min_width or height < self.min_height:
                result["valid"] = False
                result["errors"].append(f"Image too small: {width}x{height}")
            
            # Check for duplicates (perceptual hash)
            img_hash = self._compute_hash(img)
            if img_hash in self.seen_hashes:
                result["valid"] = False
                result["errors"].append("Duplicate image")
            else:
                self.seen_hashes.add(img_hash)
            
            result["width"] = width
            result["height"] = height
            result["size_kb"] = file_size / 1024
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Error: {str(e)}")
        
        return result
    
    def _compute_hash(self, img: np.ndarray) -> str:
        """Compute perceptual hash for duplicate detection"""
        # Resize to 8x8 for quick comparison
        resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Compute hash
        avg = gray.mean()
        diff = gray > avg
        hash_str = ''.join(str(int(x)) for x in diff.flatten())
        
        return hash_str
    
    def validate_batch(self, image_paths: List[Path]) -> Dict[str, any]:
        """Validate batch of images"""
        
        results = {
            "total": len(image_paths),
            "valid": 0,
            "invalid": 0,
            "details": []
        }
        
        for img_path in image_paths:
            validation = self.validate_image(img_path)
            results["details"].append(validation)
            
            if validation["valid"]:
                results["valid"] += 1
            else:
                results["invalid"] += 1
        
        return results


if __name__ == "__main__":
    # Example usage
    validator = ImageValidator(min_width=50, min_height=50)
    
    # Validate single image
    result = validator.validate_image(Path("sample.jpg"))
    print(result)

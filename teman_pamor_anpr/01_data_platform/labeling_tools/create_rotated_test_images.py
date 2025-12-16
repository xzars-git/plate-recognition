"""
Create Rotated Test Images
Generate 90°, 180°, 270° rotated versions of sample images for testing rotation handling
"""

import cv2
from pathlib import Path
import sys


def create_rotated_versions(image_path, output_dir="test_rotated"):
    """
    Create rotated versions (90°, 180°, 270°) of an image
    
    Args:
        image_path: Path to input image
        output_dir: Output directory for rotated images
    """
    print(f"📷 Loading image: {image_path}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot read image: {image_path}")
        return
    
    # Get base name
    base_name = Path(image_path).stem
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save original
    original_file = output_path / f"{base_name}_original.jpg"
    cv2.imwrite(str(original_file), image)
    print(f"✅ Original saved: {original_file}")
    
    # Rotate 90° clockwise
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rot90_file = output_path / f"{base_name}_rot90.jpg"
    cv2.imwrite(str(rot90_file), rotated_90)
    print(f"✅ 90° rotation saved: {rot90_file}")
    
    # Rotate 180°
    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
    rot180_file = output_path / f"{base_name}_rot180.jpg"
    cv2.imwrite(str(rot180_file), rotated_180)
    print(f"✅ 180° rotation saved: {rot180_file}")
    
    # Rotate 270° clockwise (90° counter-clockwise)
    rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rot270_file = output_path / f"{base_name}_rot270.jpg"
    cv2.imwrite(str(rot270_file), rotated_270)
    print(f"✅ 270° rotation saved: {rot270_file}")
    
    print(f"\n📁 All rotated images saved to: {output_path}/")
    print(f"\n🧪 To test rotation handling:")
    print(f"   python test_epoch170_with_rotation.py {rot90_file}")
    print(f"   python test_epoch170_with_rotation.py {rot180_file}")
    print(f"   python test_epoch170_with_rotation.py {rot270_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Default: use first validation image
        default_image = "dataset/plate_detection_yolo/images/val/H2359SV_jpg.rf.9ea5e8f81410cd07fc844e96f38e61cd.jpg"
        
        if Path(default_image).exists():
            print("No image specified, using default validation image")
            create_rotated_versions(default_image)
        else:
            print("Usage: python create_rotated_test_images.py <image_path>")
            print("\nExample:")
            print("  python create_rotated_test_images.py dataset/plate_detection_yolo/images/val/H2359SV_jpg.rf.9ea5e8f81410cd07fc844e96f38e61cd.jpg")
    else:
        image_path = sys.argv[1]
        create_rotated_versions(image_path)

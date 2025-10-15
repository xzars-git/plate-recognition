"""
Simple Model Export - Direct YOLO export tanpa dependency issues
"""

from ultralytics import YOLO
from pathlib import Path
import sys

def export_model(model_path='best.pt', formats=['onnx']):
    """
    Export model to various formats using YOLO built-in export
    
    Formats yang didukung:
    - onnx: Cross-platform (RECOMMENDED)
    - torchscript: PyTorch format
    - coreml: iOS/macOS
    - tflite: Mobile/Edge (need TensorFlow)
    """
    
    print("="*70)
    print("🔄 YOLO Model Export")
    print("="*70)
    
    if not Path(model_path).exists():
        print(f"\n❌ Model not found: {model_path}")
        return
    
    print(f"\n📦 Loading: {model_path}")
    model = YOLO(model_path)
    print("   ✅ Loaded!")
    
    results = {}
    
    for fmt in formats:
        print(f"\n{'='*70}")
        print(f"📤 Exporting to {fmt.upper()}...")
        print(f"{'='*70}")
        
        try:
            # Export settings per format
            export_args = {
                'format': fmt,
                'imgsz': 640,
            }
            
            # Format-specific settings
            if fmt == 'onnx':
                export_args['simplify'] = True
                export_args['opset'] = 12
            elif fmt == 'tflite':
                export_args['int8'] = False  # No quantization (easier)
            elif fmt == 'coreml':
                export_args['nms'] = True
            
            # Export
            export_path = model.export(**export_args)
            
            # Get file size
            file_path = Path(export_path)
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                results[fmt] = {
                    'status': '✅',
                    'path': str(file_path),
                    'size': size_mb
                }
                print(f"\n✅ Success!")
                print(f"   File: {file_path}")
                print(f"   Size: {size_mb:.2f} MB")
            else:
                results[fmt] = {'status': '❌', 'error': 'File not found after export'}
                
        except Exception as e:
            error_msg = str(e)
            results[fmt] = {'status': '❌', 'error': error_msg}
            print(f"\n❌ Failed!")
            print(f"   Error: {error_msg[:100]}")
            
            # Specific error messages
            if 'tensorflow' in error_msg.lower():
                print(f"\n   💡 To fix: pip install tensorflow")
            elif 'onnx2tf' in error_msg.lower():
                print(f"\n   💡 To fix: pip install onnx2tf onnx")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Export Summary")
    print(f"{'='*70}")
    
    for fmt, result in results.items():
        status = result['status']
        if status == '✅':
            print(f"{status} {fmt.upper():12} → {result['size']:.2f} MB ({result['path']})")
        else:
            print(f"{status} {fmt.upper():12} → {result['error'][:50]}")
    
    print(f"{'='*70}")
    
    # Success count
    success = sum(1 for r in results.values() if r['status'] == '✅')
    print(f"\n✅ Successfully exported: {success}/{len(formats)} formats")
    
    return results


def main():
    """Main function"""
    
    print("\n🎯 Quick Export Tool")
    print("\nAvailable formats:")
    print("  1. ONNX (recommended)")
    print("  2. TorchScript")
    print("  3. CoreML (iOS)")
    print("  4. TFLite (mobile)")
    print("  5. All formats")
    
    choice = input("\nSelect format (1-5) [default=1]: ").strip() or "1"
    
    format_map = {
        '1': ['onnx'],
        '2': ['torchscript'],
        '3': ['coreml'],
        '4': ['tflite'],
        '5': ['onnx', 'torchscript', 'tflite', 'coreml']
    }
    
    formats = format_map.get(choice, ['onnx'])
    
    # Export
    export_model('best.pt', formats)
    
    print("\n🚀 Done!")
    print("\nNext steps:")
    print("  - Test exported model")
    print("  - Deploy to target platform")
    print("  - Check CONVERSION_GUIDE.md for usage examples")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Command line: python simple_export.py onnx tflite
        formats = sys.argv[1:]
        export_model('best.pt', formats)
    else:
        # Interactive mode
        main()

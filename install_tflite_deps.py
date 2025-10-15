"""
Install TFLite Dependencies
Script untuk install semua dependency yang diperlukan untuk TFLite conversion
"""

import subprocess
import sys

def install_packages():
    """
    Install all required packages for TFLite conversion
    """
    
    print("="*70)
    print("📦 TFLite Dependency Installer")
    print("="*70)
    
    packages = [
        ('tensorflow', 'TensorFlow framework'),
        ('tf-keras', 'Keras for TensorFlow 2.x'),
        ('onnx', 'ONNX format support'),
        ('onnx2tf', 'ONNX to TensorFlow converter'),
        ('onnx-graphsurgeon', 'ONNX graph optimization'),
    ]
    
    print("\n📋 Packages to install:")
    for pkg, desc in packages:
        print(f"   - {pkg:20} ({desc})")
    
    print(f"\n⚠️ Warning:")
    print(f"   Total download size: ~500MB")
    print(f"   Installation time: 5-10 minutes")
    
    choice = input("\nProceed with installation? (y/n): ").lower()
    if choice != 'y':
        print("\n❌ Installation cancelled")
        print("\n💡 Alternative: Use ONNX format instead")
        print("   - Already exported: best.onnx")
        print("   - No heavy dependencies needed")
        print("   - Test with: python test_onnx.py")
        return False
    
    print("\n" + "="*70)
    print("📥 Installing packages...")
    print("="*70)
    
    success = []
    failed = []
    
    for pkg, desc in packages:
        print(f"\n📦 Installing {pkg}...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                pkg, '--quiet'
            ])
            success.append(pkg)
            print(f"   ✅ {pkg} installed successfully")
        except subprocess.CalledProcessError:
            failed.append(pkg)
            print(f"   ❌ {pkg} installation failed")
    
    # Summary
    print("\n" + "="*70)
    print("📊 Installation Summary")
    print("="*70)
    
    print(f"\n✅ Successful ({len(success)}/{len(packages)}):")
    for pkg in success:
        print(f"   ✅ {pkg}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}/{len(packages)}):")
        for pkg in failed:
            print(f"   ❌ {pkg}")
        
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check internet connection")
        print(f"   2. Try manual install: pip install {' '.join(failed)}")
        print(f"   3. Check pip version: python -m pip --version")
        return False
    else:
        print(f"\n{'='*70}")
        print("✅ All dependencies installed successfully!")
        print(f"{'='*70}")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Convert to TFLite:")
        print(f"      python convert_to_tflite.py")
        print(f"   2. Or use ONNX (already exported):")
        print(f"      python test_onnx.py")
        
        return True


def check_installation():
    """
    Check if all dependencies are installed
    """
    
    print("="*70)
    print("🔍 Checking TFLite Dependencies")
    print("="*70)
    
    packages = {
        'tensorflow': 'TensorFlow',
        'tf_keras': 'tf-keras',
        'onnx': 'ONNX',
        'onnx2tf': 'onnx2tf',
        'onnx_graphsurgeon': 'onnx-graphsurgeon',
    }
    
    installed = []
    missing = []
    
    for module, name in packages.items():
        try:
            __import__(module)
            installed.append(name)
            print(f"   ✅ {name}")
        except ImportError:
            missing.append(name)
            print(f"   ❌ {name} (not installed)")
    
    print(f"\n{'='*70}")
    if missing:
        print(f"❌ Missing {len(missing)} package(s)")
        print(f"{'='*70}")
        print(f"\nMissing packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        
        print(f"\n💡 To install missing packages:")
        print(f"   python install_tflite_deps.py")
        return False
    else:
        print(f"✅ All dependencies installed!")
        print(f"{'='*70}")
        print(f"\n🚀 Ready to convert:")
        print(f"   python convert_to_tflite.py")
        return True


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        check_installation()
    else:
        install_packages()

"""
Automatic PyTorch CUDA Installation Script
For NVIDIA RTX 3080 Ti
"""

import subprocess
import sys

print("="*70)
print("🎮 PyTorch CUDA Installation for RTX 3080 Ti")
print("="*70)

# Check current PyTorch
print("\n📦 Checking current PyTorch installation...")
try:
    import torch
    print(f"   Current PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ Already using GPU! No need to reinstall.")
        sys.exit(0)
    else:
        print("   ⚠️  CUDA not available - need to reinstall PyTorch")
except ImportError:
    print("   PyTorch not installed")

# Uninstall current PyTorch
print("\n🗑️  Uninstalling current PyTorch...")
subprocess.run([
    sys.executable, '-m', 'pip', 'uninstall', 
    'torch', 'torchvision', 'torchaudio', '-y'
], check=False)

# Install PyTorch with CUDA 12.1 (best for RTX 3080 Ti)
print("\n📥 Installing PyTorch with CUDA 12.1...")
print("   (This will take a few minutes, ~2GB download)")
print("="*70)

result = subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'torch', 'torchvision', 'torchaudio',
    '--index-url', 'https://download.pytorch.org/whl/cu121'
], check=False)

if result.returncode != 0:
    print("\n⚠️  CUDA 12.1 installation failed, trying CUDA 11.8...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/cu118'
    ], check=True)

# Reinstall Ultralytics
print("\n📥 Reinstalling Ultralytics...")
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--upgrade', 'ultralytics'
], check=True)

# Verify installation
print("\n" + "="*70)
print("✅ Installation Complete! Verifying...")
print("="*70)

import torch
print(f"\n📦 PyTorch Version: {torch.__version__}")
print(f"🔥 CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"🎮 GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"⚡ CUDA Version: {torch.version.cuda}")
    print("\n" + "="*70)
    print("🎉 SUCCESS! Your RTX 3080 Ti is ready for training!")
    print("="*70)
    print("\n💡 Tips for RTX 3080 Ti:")
    print("   - Increase batch size to 16 or 32 (you have 12GB VRAM!)")
    print("   - Set workers=4 or 8 for faster data loading")
    print("   - Enable amp=True for mixed precision (faster training)")
    print("   - Change optimizer to 'auto' or 'AdamW'")
    print("\n📝 Update train_plate_detection.py:")
    print("   epochs=150")
    print("   batch=16        # or 32 if fits in memory")
    print("   workers=4       # faster data loading")
    print("   amp=True        # mixed precision")
    print("   optimizer='auto' # AdamW optimizer")
else:
    print("\n❌ CUDA still not available!")
    print("\n🔧 Troubleshooting:")
    print("   1. Check NVIDIA Driver:")
    print("      Run: nvidia-smi")
    print("   2. Update NVIDIA Driver:")
    print("      https://www.nvidia.com/Download/index.aspx")
    print("   3. Restart PC and try again")

print("="*70)

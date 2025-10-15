# Activate Virtual Environment dan Install Dependencies
# PowerShell Script

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*59) -ForegroundColor Cyan
Write-Host "🚀 ANPR Setup - Virtual Environment" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*59) -ForegroundColor Cyan

# 1. Activate venv
Write-Host "`n📦 Step 1: Activating Virtual Environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Virtual environment activated!`n" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to activate venv. Creating new one...`n" -ForegroundColor Red
    python -m venv venv
    & ".\venv\Scripts\Activate.ps1"
}

# 2. Check if packages are installed
Write-Host "📦 Step 2: Checking installed packages..." -ForegroundColor Yellow

$packages = @("ultralytics", "torch", "opencv-python", "paddleocr")
$needInstall = $false

foreach ($package in $packages) {
    $installed = pip show $package 2>$null
    if ($installed) {
        Write-Host "  ✅ $package is installed" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $package is NOT installed" -ForegroundColor Red
        $needInstall = $true
    }
}

# 3. Install if needed
if ($needInstall) {
    Write-Host "`n📥 Step 3: Installing missing packages..." -ForegroundColor Yellow
    Write-Host "This may take 5-10 minutes...`n" -ForegroundColor Cyan
    
    pip install ultralytics torch torchvision opencv-python paddlepaddle paddleocr pandas --index-url https://download.pytorch.org/whl/cpu
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ All packages installed successfully!" -ForegroundColor Green
    } else {
        Write-Host "`n❌ Installation failed. Please check errors above." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n✅ All required packages are already installed!" -ForegroundColor Green
}

# 4. Verify installation
Write-Host "`n🧪 Step 4: Verifying installation..." -ForegroundColor Yellow

python -c "from ultralytics import YOLO; print('  ✅ Ultralytics OK')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Ultralytics verification failed" -ForegroundColor Red
}

python -c "import torch; print('  ✅ PyTorch OK')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ PyTorch verification failed" -ForegroundColor Red
}

python -c "import cv2; print('  ✅ OpenCV OK')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ OpenCV verification failed" -ForegroundColor Red
}

# 5. Show next steps
Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*59) -ForegroundColor Cyan
Write-Host "🎯 READY TO GO!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*59) -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Test training:    python test_training.py" -ForegroundColor Cyan
Write-Host "  2. Full training:    python train_plate_detection.py" -ForegroundColor Cyan
Write-Host "  3. Check dataset:    python check_dataset.py" -ForegroundColor Cyan
Write-Host "`n💡 Virtual environment is ACTIVE in this terminal!" -ForegroundColor Green
Write-Host ""

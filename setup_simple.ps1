# Simple Setup Script
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "ANPR Setup - Activating Environment" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan

# Activate venv
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

Write-Host "✅ Virtual environment activated!`n" -ForegroundColor Green

# Check ultralytics
Write-Host "Checking if ultralytics is installed..." -ForegroundColor Yellow
$check = python -c "import ultralytics; print('installed')" 2>&1

if ($check -match "installed") {
    Write-Host "✅ Ultralytics is installed`n" -ForegroundColor Green
    
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host "Ready! You can now run:" -ForegroundColor Green
    Write-Host "  python test_training.py" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
} else {
    Write-Host "❌ Ultralytics not found`n" -ForegroundColor Red
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    Write-Host "This will take 5-10 minutes...`n" -ForegroundColor Cyan
    
    pip install ultralytics opencv-python pandas
    
    Write-Host "`n✅ Installation complete!" -ForegroundColor Green
    Write-Host "`nYou can now run:" -ForegroundColor Green
    Write-Host "  python test_training.py" -ForegroundColor Cyan
}

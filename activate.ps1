# HybridMind Development Environment
# Python 3.11 with CUDA Support

Write-Host "🚀 Activating HybridMind Dev Environment..." -ForegroundColor Cyan
.\venv311\Scripts\Activate.ps1
Write-Host "✅ Environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "GPU Info:" -ForegroundColor Yellow
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
Write-Host ""
Write-Host "Quick Commands:" -ForegroundColor Yellow
Write-Host "  uvicorn main:app --reload  → Start API server"
Write-Host "  streamlit run ui/app.py    → Start UI"
Write-Host "  pytest                     → Run tests"
Write-Host ""


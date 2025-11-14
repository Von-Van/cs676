# TinyTroupe Persona Simulator - Production Version Launch Script
# Enhanced PowerShell launcher with comprehensive error handling

# Set script execution policy for this session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# Change to the app directory
Set-Location -Path $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TinyTroupe Simulation Engine v2.0" -ForegroundColor White
Write-Host "  Production-Ready Persona Simulator" -ForegroundColor White  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check and activate virtual environment
$venvPaths = @(".venv\Scripts\Activate.ps1", "..\..\..venv\Scripts\Activate.ps1")
$venvActivated = $false

foreach ($venvPath in $venvPaths) {
    if (Test-Path $venvPath) {
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        & $venvPath
        $venvActivated = $true
        break
    }
}

if (-not $venvActivated) {
    Write-Host "No virtual environment found - using system Python" -ForegroundColor Yellow
}

# Check if Streamlit is available
try {
    streamlit --version | Out-Null
    Write-Host "Streamlit found" -ForegroundColor Green
} catch {
    Write-Host "Streamlit not found - installing requirements..." -ForegroundColor Red
    pip install -r requirements.txt
}

# Check for API key
if (-not $env:OPENAI_API_KEY) {
    Write-Host "Warning: OPENAI_API_KEY environment variable not set" -ForegroundColor Yellow
    Write-Host "Make sure to configure your API key in the Streamlit app" -ForegroundColor Yellow
}

# Launch the enhanced Streamlit application
Write-Host ""
Write-Host "Launching TinyTroupe Production Interface..." -ForegroundColor Green
Write-Host "Browser will open automatically at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Features available:" -ForegroundColor White
Write-Host "  • Enhanced persona validation" -ForegroundColor Gray
Write-Host "  • Real-time performance monitoring" -ForegroundColor Gray
Write-Host "  • Advanced conversation analytics" -ForegroundColor Gray
Write-Host "  • Production-grade error handling" -ForegroundColor Gray
Write-Host ""

try {
    streamlit run app_updated.py
    Write-Host "Application closed normally." -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERROR: Failed to launch application" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Possible solutions:" -ForegroundColor Yellow
    Write-Host "1. Run 'pip install -r requirements.txt'" -ForegroundColor White
    Write-Host "2. Set your OPENAI_API_KEY environment variable" -ForegroundColor White
    Write-Host "3. Check that Python and Streamlit are installed" -ForegroundColor White
    Write-Host "4. Try running 'python scripts/simulation_engine.py --help' for CLI mode" -ForegroundColor White
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

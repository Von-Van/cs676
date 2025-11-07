# Launch TinyTroupe Persona Simulator
# Change to the app directory
Set-Location -Path $PSScriptRoot

# Activate virtual environment if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
}

# Launch Streamlit
Write-Host "Starting TinyTroupe Persona Simulator..." -ForegroundColor Green
streamlit run app.py

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError launching app. Press any key to exit." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

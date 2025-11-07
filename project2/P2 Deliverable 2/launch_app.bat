@echo off
REM Launch TinyTroupe Persona Simulator
echo Starting TinyTroupe Persona Simulator...
echo.

REM Change to the app directory
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Launch Streamlit
streamlit run app.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Error launching app. Press any key to exit.
    pause > nul
)

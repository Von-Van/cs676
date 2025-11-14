@echo off
REM Launch TinyTroupe Persona Simulator - Production Version
echo ========================================
echo  TinyTroupe Simulation Engine v2.0
echo  Production-Ready Persona Simulator  
echo ========================================
echo.
echo Starting application...
echo.

REM Change to the app directory
cd /d "%~dp0"

REM Check if virtual environment exists and activate
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else if exist "..\..\..venv\Scripts\activate.bat" (
    echo Activating project virtual environment...
    call "..\..\..venv\Scripts\activate.bat"
)

REM Check if streamlit is available
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Streamlit not found. Installing requirements...
    pip install -r requirements.txt
)

REM Launch the enhanced Streamlit app
echo Launching TinyTroupe Production Interface...
echo Browser will open automatically at http://localhost:8501
echo.
streamlit run app_updated.py

REM Handle errors gracefully
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Failed to launch application
    echo ========================================
    echo Possible solutions:
    echo 1. Run 'pip install -r requirements.txt'
    echo 2. Set your OPENAI_API_KEY environment variable
    echo 3. Check that Python and Streamlit are installed
    echo.
    echo Press any key to exit...
    pause > nul
) else (
    echo.
    echo Application closed normally.
    timeout /t 3 > nul
)

# TinyTroupe Persona Simulator - Quick Launch Guide

## Option 1: Double-Click Launch (Recommended)

### Windows Batch File (Easiest):
1. Double-click `launch_app.bat` in this folder
2. A browser window will open automatically with the app
3. Close the command window when done

### PowerShell (Alternative):
1. Right-click `launch_app.ps1` → "Run with PowerShell"
2. If you get a security warning, run this once in PowerShell as Administrator:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

## Option 2: Create Desktop Shortcut

### For Batch File:
1. Right-click `launch_app.bat`
2. Click "Create shortcut"
3. Drag the shortcut to your Desktop
4. (Optional) Right-click shortcut → Properties → Change Icon to customize

### For PowerShell:
1. Right-click Desktop → New → Shortcut
2. Enter location: 
   ```
   powershell.exe -ExecutionPolicy Bypass -File "C:\Users\jakem\Documents\GitHub\cs676\project2\P2 Deliverable 2\launch_app.ps1"
   ```
3. Name it "TinyTroupe Simulator"
4. (Optional) Right-click → Properties → Change Icon

## Option 3: Pin to Taskbar
1. Create a desktop shortcut (see above)
2. Right-click the shortcut → "Pin to taskbar"
3. Now you can launch with one click from taskbar!

## Option 4: Start Menu Entry
1. Press `Win + R`
2. Type: `shell:startup`
3. Copy `launch_app.bat` to this folder
4. It will now appear in your Start Menu

## Manual Launch (If scripts don't work):
```cmd
cd ""
streamlit run app.py
```

## Troubleshooting:
- **"streamlit is not recognized"**: Virtual environment not activated
- **Import errors**: Run `pip install -r requirements.txt`
- **Port already in use**: Close other Streamlit instances or use `streamlit run app.py --server.port 8502`

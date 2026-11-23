@echo off
rem Lance Volumeo dans sa fenetre.
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
  echo Volumeo n est pas encore installe : lancez d abord "Installer Volumeo.bat".
  pause
  exit /b 1
)
cd /d "%~dp0backend"
start "" "%~dp0.venv\Scripts\pythonw.exe" -m app.desktop

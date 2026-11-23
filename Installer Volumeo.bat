@echo off
rem Double-cliquez pour installer Volumeo (ou le mettre a jour).
title Installation de Volumeo
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows\install.ps1" %*
if errorlevel 1 (
  echo.
  echo L installation a echoue. Le message ci-dessus indique pourquoi.
)
echo.
pause

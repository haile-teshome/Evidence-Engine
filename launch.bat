@echo off
REM Windows launcher for Evidence Engine. Double-click to run.
REM Self-locating via %~dp0; works wherever this folder lives.
title Evidence Engine

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is required. Opening the download page...
  start "" "https://nodejs.org/en/download"
  echo Install Node.js LTS, then double-click this file again.
  pause
  exit /b 1
)

node "%~dp0launch.mjs"
echo.
echo Evidence Engine has stopped. You can close this window.
pause >nul

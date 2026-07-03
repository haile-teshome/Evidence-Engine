@echo off
REM Windows launcher for Evidence Engine. Double-click to run.
REM Keep this file in the project root; %~dp0 resolves to the project folder.
title Evidence Engine
node "%~dp0launch.mjs"
echo.
echo Evidence Engine has stopped. You can close this window.
pause >nul

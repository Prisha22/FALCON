@echo off
REM FALCON setup launcher.
REM PowerShell scripts are blocked by default execution policy on a fresh
REM Windows install, so this bypasses it for this one invocation only --
REM it does not change any machine setting.
REM
REM   setup.bat              install for the current user
REM   setup.bat -SystemPath  machine-wide PATH (run this file as Administrator)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*

echo.
pause

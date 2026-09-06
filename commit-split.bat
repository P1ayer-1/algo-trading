@echo off
cd /d "%~dp0"
echo Staging and committing the module split ...
git add -A
git commit -m "Split live-chart backend and frontend into focused modules"
echo.
echo Done.
pause

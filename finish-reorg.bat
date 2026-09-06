@echo off
setlocal

rem Run this AFTER Claude has already added backend\, frontend\, README.md
rem and the updated .gitignore to this folder. This script only cleans up
rem the old root-level copies and finishes the move of node_modules, then
rem commits everything to git.

cd /d "%~dp0"

echo Moving node_modules into frontend\ ...
if exist node_modules (
    if exist frontend\node_modules (
        echo   frontend\node_modules already exists - removing old root copy instead.
        rmdir /s /q node_modules
    ) else (
        move node_modules frontend\node_modules
    )
) else (
    echo   No root-level node_modules found, skipping.
)

echo Removing old root-level files that were moved into backend\/frontend\ ...
if exist live-chart.py del /q live-chart.py
if exist live-chart.html del /q live-chart.html
if exist package.json del /q package.json
if exist package-lock.json del /q package-lock.json

echo Clearing stale __pycache__ (it referenced the old file locations) ...
if exist __pycache__ rmdir /s /q __pycache__

echo.
echo Staging and committing with git ...
git add -A
git commit -m "Reorganize project into backend/ and frontend/"

echo.
echo Done. Run "git status" to double-check, and "cd frontend && npm install"
echo if frontend\node_modules did not survive the move.
pause

@echo off
title Biometric Face Attendance System
echo ========================================================
echo   Biometric Attendance Management System (BioAttendAI)
echo   Computer Vision Engine: YuNet + SFace (128D) + FER+
echo ========================================================
echo.

IF NOT EXIST "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .\venv!
    echo Please create the virtual environment and install requirements:
    echo    python -m venv venv
    echo    .\venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting Flask Application Server...
echo.
echo   Admin Dashboard:    http://127.0.0.1:5000/admin/login
echo   Student Portal:     http://127.0.0.1:5000/attendance
echo   Default Admin:      admin / admin123
echo.
.\venv\Scripts\python.exe app.py
pause


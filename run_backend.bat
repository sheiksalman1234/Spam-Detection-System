@echo off
title SpamShield AI - Backend Server
color 0A

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║         SpamShield AI - Backend Server                     ║
echo ║         Powered by Gemini API Translation                  ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

cd /d c:\SpamCallDetection\backend

echo [*] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [!] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo [✓] Python found
echo.

echo [*] Installing/Updating dependencies...
echo     - fastapi, uvicorn
echo     - torch, transformers
echo     - openai-whisper
echo     - librosa
echo     - google-generativeai
echo.

pip install -q -r requirements.txt
if errorlevel 1 (
    echo [!] Failed to install dependencies
    pause
    exit /b 1
)

echo [✓] Dependencies installed
echo.

echo [*] Starting SpamShield Backend...
echo     - Gemini API Key: Configured
echo     - Supported Languages: EN, HI, TE, TA, KN, ML
echo     - Server: http://localhost:8000
echo.

python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause

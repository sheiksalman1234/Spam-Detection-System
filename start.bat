@echo off
cd /d C:\SpamCallDetection\backend
C:\Users\salma\spamvenv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause

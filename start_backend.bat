@echo off
cd backend
..\venv\Scripts\pythonw.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --workers 2 > ..\.codex-runlogs\backend-8002.log 2> ..\.codex-runlogs\backend-8002.err.log

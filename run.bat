@echo off
title AstroCommander OS - Tri-Hazard Flight Deck
echo ============================================================
echo   ASTROCOMMANDER OS // TRI-HAZARD DECISION ENGINE
echo   Unified Spacecraft Multi-Phase Fire Containment Suite
echo   BASS-II (Solid) | FLEX (Liquid) | CFI-G (Gas)
echo ============================================================
echo Starting FastAPI Server on http://127.0.0.1:8080 ...
echo.

start "" "http://127.0.0.1:8080"
python -m uvicorn server:app --host 127.0.0.1 --port 8080 --reload

pause

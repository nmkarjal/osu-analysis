@echo off
rem Double-click after a session: pull, analyse, plan.
cd /d "%~dp0"
python -m osu_coach run %*
pause

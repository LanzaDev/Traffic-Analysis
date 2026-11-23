@echo off
REM Launcher do projeto de Analise de Trafego (usa o venv local).
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [erro] venv nao encontrado. Crie com: python -m venv venv
    echo        e instale: venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)
venv\Scripts\python.exe main.py %*

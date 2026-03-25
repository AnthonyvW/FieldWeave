@echo off
setlocal

echo Starting FieldWeave...

if exist venv\ (
    echo Virtual environment found. Activating...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements.
        pause
        exit /b 1
    )
)

echo Launching FieldWeave...
python main.py

endlocal

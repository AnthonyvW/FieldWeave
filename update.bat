@echo off
setlocal

echo Updating FieldWeave...

echo Pulling latest changes from git...
git pull
if errorlevel 1 (
    echo ERROR: git pull failed. Check your connection or repository status.
    pause
    exit /b 1
)

if not exist venv\ (
    echo Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing/updating dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

echo FieldWeave updated successfully.
pause
endlocal

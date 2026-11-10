@echo off
setlocal enabledelayedexpansion

echo Checking for compatible Python installation...

:: 1. Check if 'py' is available
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python launcher 'py' was not found on this computer.
    echo Please install Python 3.11 - 3.13 from https://www.python.org/downloads/
    echo Make sure to check "Add python.exe to PATH" and "Install Python Launcher".
    echo.
    pause
    exit /b 1
)

:: 2. Get the default Python version (e.g., "Python 3.14.7") and extract the minor version
for /f "tokens=2" %%I in ('py --version 2^>^&1') do set FULL_VERSION=%%I
for /f "tokens=2 delims=." %%I in ("%FULL_VERSION%") do set MINOR_VERSION=%%I

:: 3. Check if the default version is in the sweet spot (3.10 - 3.13)
if !MINOR_VERSION! geq 10 (
    if !MINOR_VERSION! leq 13 (
        echo Found compatible default version: Python 3.!MINOR_VERSION!
        set "PY_CMD=py"
        goto :setup_env
    )
)

:: 4. If default is wrong (e.g., 3.14 or 3.9), search for other compatible installed versions
echo Default Python (3.%MINOR_VERSION%) is not recommended for this AI setup.
echo Searching for existing compatible versions (3.13 down to 3.10)...

for %%V in (13 12 11 10) do (
    py -3.%%V --version >nul 2>&1
    if !errorlevel! equ 0 (
        echo Python 3.%%V found! Switching context...
        set "PY_CMD=py -3.%%V"
        goto :setup_env
    )
)

:: 5. If no compatible versions are found, try to auto-install 3.13
echo No compatible versions found locally. Attempting auto-installation of Python 3.13...
py install 3.13
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to auto-install Python 3.13. 
    echo Please manually install Python 3.13 from python.org
    pause
    exit /b 1
)

:: Verify the install worked
py -3.13 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Successfully installed and switched to Python 3.13!
    set "PY_CMD=py -3.13"
    goto :setup_env
) else (
    echo [ERROR] Installation appeared to work but 3.13 is still missing.
    pause
    exit /b 1
)

:setup_env
echo.
echo ========================================================
echo Setting up the Text-to-Speech Environment...
echo Using command: %PY_CMD%
echo ========================================================
echo.

%PY_CMD% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)

echo.
echo Generating offline audio previews...
%PY_CMD% generate_samples.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Sample generation failed.
    pause
    exit /b 1
)

:: Save the successful command to the marker file for run.bat
echo %PY_CMD%> .setup_done
echo.
echo Setup Complete! You can now use run.bat to start the app.
pause
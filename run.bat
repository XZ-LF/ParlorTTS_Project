@echo off

:: 1. If setup hasn't run, trigger it
if not exist .setup_done (
    echo ========================================================
    echo  [NOTICE] First-time setup required.
    echo  Running setup.bat automatically... Please wait.
    echo ========================================================
    echo.
    call setup.bat
    
    :: Verify it succeeded
    if not exist .setup_done (
        echo.
        echo [ERROR] Setup did not complete successfully. Aborting.
        pause
        exit /b 1
    )
)

:: 2. Read the correct Python command from the marker file
set /p PY_CMD=<.setup_done

:: 3. Launch the app using the saved command
echo Starting the Expressive TTS System...
%PY_CMD% app.py %*

pause
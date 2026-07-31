@rem JMod Windows Launcher
@rem ---------------------------------------------------------------------------------------------------------------------------------------

@rem Turn off command echoing so terminal stays clean
@echo off

@rem "setlocal" keeps environment variable changes local to this script only, once the script exits, variables like PATH revert back
if "%OS%"=="Windows_NT" setlocal

@rem Set directory where this .bat file lives
set DIRNAME=%~dp0

@rem If for some reason DIRNAME is empty, use current directory instead
if "%DIRNAME%"=="" set DIRNAME=.

@rem Convert path into full absolute path
for %%i in ("%DIRNAME%") do set APP_HOME=%%~fi

@rem Change terminal into JMod directory (if needed)
cd /d "%APP_HOME%"

@rem ---------------------------------------------------------------------------------------------------------------------------------------
s
@rem Check if UV already exists on system, if there is no error jump down to uvFound (no need to install UV again)
where uv >nul 2>&1
if %ERRORLEVEL% equ 0 goto uvFound

@rem ---------------------------------------------------------------------------------------------------------------------------------------
@rem Check whether pip exists, if it does not skip to curl/powershell installer
where pip >nul 2>&1
if %ERRORLEVEL% neq 0 goto tryInstaller

@rem Install UV via pip
pip install uv

@rem Check whether install succeeded, if so jump down to uvFound
where uv >nul 2>&1
if %ERRORLEVEL% equ 0 goto uvFound

@rem ---------------------------------------------------------------------------------------------------------------------------------------
@rem Try downloading UV with the official installer
:tryInstaller

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

@rem Add UV Installation location to PATH for this session
set PATH=%USERPROFILE%\.local\bin;%PATH%

@rem Check again whether UV now exists, if still missing go to uvFailed
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 goto uvFailed

@rem ---------------------------------------------------------------------------------------------------------------------------------------
@rem If UV successfully found, set up environment
:uvFound

@rem Sync UV environment
uv sync --python 3.11

@rem Launch GUI application
start "" uv run pythonw run_jmod_from_GUI.py

@rem Done 
goto end

@rem ---------------------------------------------------------------------------------------------------------------------------------------
@rem If UV Installation failed:
:uvFailed

echo ERROR: Failed to install UV automatically, please install UV manually

pause

goto end

@rem ---------------------------------------------------------------------------------------------------------------------------------------
@rem Cleanup 
:end

@rem Restore original environment variables before exiting
if "%OS%"=="Windows_NT" endlocal


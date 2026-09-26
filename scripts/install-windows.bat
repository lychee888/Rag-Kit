@echo off
setlocal enabledelayedexpansion
title rag-kit Installer (Windows)

REM ============================================================
REM   rag-kit Installer - Windows
REM   Agentic RAG System for Hermes Agent
REM   - installs all dependencies (pyproject: .[ocr])
REM   - installs & boots the state-change watcher
REM   - optional pinned model pre-download (offline-ready)
REM ============================================================
echo.
echo ============================================================
echo   rag-kit Installer - Windows
echo   Agentic RAG System for Hermes Agent
echo ============================================================
echo.

set "INSTALL_DIR=%USERPROFILE%\rag-kit-venv"
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "WATCH_FOLDER=%USERPROFILE%\Documents\rag-ingest"

REM ---- Step 1: Check Python >= 3.10 ----
echo [1/8] Checking Python...

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: Python not found in PATH. Please install Python 3.10+ first.
    echo   Download: https://www.python.org/downloads/
    goto :end_fail
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo   Found Python %PY_VER%

for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

if !PY_MAJOR! LSS 3 (
    echo   ERROR: Python 3.10+ required, found %PY_VER%
    goto :end_fail
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 (
    echo   ERROR: Python 3.10+ required, found %PY_VER%
    goto :end_fail
)
echo   OK: Python %PY_VER% meets requirements
echo.

REM ---- Step 2: Detect GPU ----
echo [2/8] Detecting GPU...

set "USE_CUDA=0"
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    nvidia-smi >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "USE_CUDA=1"
        for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do set "GPU_NAME=%%g"
        for /f "tokens=*" %%c in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do set "GPU_CC=%%c"
        echo   GPU detected: !GPU_NAME! ^(compute cap !GPU_CC!^)
        echo   Will install a PyTorch build compatible with this GPU.
    ) else (
        echo   nvidia-smi found but GPU not available. Using CPU mode.
    )
) else (
    echo   No NVIDIA GPU detected. Using CPU mode.
)
echo.

REM ---- Step 3: Create virtual environment ----
echo [3/8] Creating virtual environment at !INSTALL_DIR!...

REM Stop any running rag-kit watcher first — it keeps the old venv's exe open,
REM which would block removing / recreating the venv (file-lock on python.exe).
taskkill /f /im rag.exe >nul 2>&1
timeout /t 1 /nobreak >nul 2>&1

if exist "!INSTALL_DIR!" (
    echo   Removing existing installation...
    rmdir /s /q "!INSTALL_DIR!" 2>nul
)

python -m venv "!INSTALL_DIR!"
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: Failed to create virtual environment.
    goto :end_fail
)
echo   Created: !INSTALL_DIR!
echo.

REM ---- Step 4: Install dependencies ----
echo [4/8] Installing dependencies...

call "!INSTALL_DIR!\Scripts\activate.bat" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: Failed to activate virtual environment.
    goto :end_fail
)

REM Upgrade pip first
python -m pip install --upgrade pip --quiet 2>nul

REM Isolate from any inherited PYTHONPATH (e.g. an agent/venv shell) so pip
REM and python resolve purely against this venv. Use a non-existent path so
REM the variable stays non-empty (an EMPTY PYTHONPATH crashes some Python builds).
set "PYTHONPATH=%TEMP%\ragkit_empty_pypath"

set "TORCH_OK="
if !USE_CUDA! EQU 1 (
    echo   Installing PyTorch ^(CUDA-enabled build — cu128, supports Blackwell^)...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
    if !ERRORLEVEL! EQU 0 set "TORCH_OK=1"
    if not defined TORCH_OK (
        echo   WARNING: cu128 build failed; trying cu124 ^(for older GPUs^)...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
        if !ERRORLEVEL! EQU 0 set "TORCH_OK=1"
    )
    if not defined TORCH_OK (
        echo   WARNING: CUDA PyTorch failed. Falling back to CPU-only PyTorch...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
        if !ERRORLEVEL! EQU 0 set "TORCH_OK=1"
    )
) else (
    echo   Installing PyTorch ^(CPU-only^)...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    if !ERRORLEVEL! EQU 0 set "TORCH_OK=1"
)

if not defined TORCH_OK (
    echo   ERROR: Failed to install PyTorch.
    goto :end_fail
)

echo   Installing rag-kit and dependencies (incl. OCR)...
cd /d "%PROJECT_DIR%"
python -m pip install -e ".[ocr]" --quiet

if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: Failed to install rag-kit.
    goto :end_fail
)
echo   Installation complete.
echo.

REM ---- Step 5: Verify imports and CLI ----
echo [5/8] Verifying installation...

REM Test imports
python -c "import rag_kit; print('  rag_kit v' + rag_kit.__version__)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   ERROR: Failed to import rag_kit.
    goto :end_fail
)

python -c "import rag_kit.autostart; print('  autostart OK')" 2>nul
python -c "import rag_kit.watchlock; print('  watchlock OK')" 2>nul

REM Heavy-stack import check (catches broken sentence-transformers / tokenizers)
python -c "import sentence_transformers, tokenizers; print('  ML stack OK (tokenizers ' + tokenizers.__version__ + ')')" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   ML stack: sentence-transformers OK
) else (
    echo   WARNING: sentence-transformers import check failed ^(dependency version conflict?^).
    echo   Reinstall with:  python -m pip install -e \".\[ocr]\"
    echo   Common cause: an incompatible tokenizers build.
)

rag --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   CLI: rag --version OK
) else (
    echo   WARNING: rag CLI not found in PATH.
    echo   You can run it from: !INSTALL_DIR!\Scripts\rag.exe
)

REM Create default config
rag config init >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   Config: Created default config
) else (
    echo   WARNING: Could not create default config ^(non-fatal^)
)

REM Create default watch folder
if not exist "%WATCH_FOLDER%" (
    mkdir "%WATCH_FOLDER%" 2>nul
    if !ERRORLEVEL! EQU 0 echo   Created watch folder: %WATCH_FOLDER%
)
echo   Verification passed.
echo.

REM ---- Step 6: Setup autostart ----
echo [6/8] Setting up autostart...

rag setup-autostart --json 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   Autostart: Installed ^(rag-kit-watcher will start on logon^)
) else (
    echo   Autostart: NOT installed ^(admin rights required for schtasks^)
    echo   To install autostart manually, run as Administrator:
    echo     "!INSTALL_DIR!\Scripts\rag.exe" setup-autostart
)
echo.

REM ---- Step 7: Initial backfill (state-change watcher takes over) ----
echo [7/8] Backfilling existing documents...

REM The autostart task runs "rag watch" on logon (event-driven).  Here we
REM backfill existing files so they are searchable immediately.  The
REM watcher uses a per-folder lock, so the one started below and the one
REM autostart launches later cannot double-ingest.
"!INSTALL_DIR!\Scripts\rag.exe" ingest "%WATCH_FOLDER%" --json >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    "!INSTALL_DIR!\Scripts\rag.exe" list-files --json 2>nul | findstr /c:"file_count" >nul
    if !ERRORLEVEL! EQU 0 (
        echo   Backfill: complete ^(documents indexed^)
    ) else (
        echo   Backfill: nothing to ingest yet ^(folder is empty; new files are picked up live^).
    )
) else (
    echo   Backfill: nothing to ingest yet ^(new files will be picked up on logon^).
)

REM Start the state-change watcher now (hidden, minimal window) so the
REM library works immediately, not only after the next logon.
start "rag-kit-watcher" /min "!INSTALL_DIR!\Scripts\rag.exe" watch "%WATCH_FOLDER%"
echo   Watcher: started in background (minimized window)
echo   Watch folder: %WATCH_FOLDER%
echo.

REM ---- Copy Hermes Agent skill ----
set "SKILL_SRC=%PROJECT_DIR%\SKILL.md"
set "SKILL_DST=%USERPROFILE%\AppData\Local\hermes\skills\research\rag-kit\SKILL.md"

if exist "%SKILL_SRC%" (
    if not exist "%SKILL_DST%" (
        mkdir "%USERPROFILE%\AppData\Local\hermes\skills\research\rag-kit" 2>nul >nul
        copy "%SKILL_SRC%" "%SKILL_DST%" >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
            echo Hermes Agent skill installed: research/rag-kit
        )
    )
)

REM ---- Step 8 (optional): Pre-download pinned models ----
echo.
echo ============================================================
echo   Optional: pre-download models for fully offline use.
echo   (embedding ~470 MB + EasyOCR ~100 MB; VLM ~330 MB optional)
echo ============================================================
set /p DL_MODELS="Pre-download pinned models now (from GitHub Releases)? [y/N]: "
if /i "!DL_MODELS!"=="y" (
    REM Download embedding + EasyOCR + SmolVLM so VLM captioning works out of the box.
    set "MODEL_DIR=%USERPROFILE%\models"
    set "SKIP_VLM=0"
    set "RAG_KIT_BATCH=1"
    call "%SCRIPT_DIR%download-models.bat"
   )

REM ---- Summary ----
echo.
echo ============================================================
echo   Installation Complete
echo ============================================================
echo.

REM Check autostart status
rag setup-autostart --json 2>nul | findstr /c:"\"installed\": true" >nul
if %ERRORLEVEL% EQU 0 (
    echo   Autostart:    Installed ^(on logon^)
) else (
    echo   Autostart:    NOT installed ^(run as Admin to install^)
)

REM Check watcher (via lock file presence is not needed now; report what we did)
echo   Watcher:      Started in background; will also start at logon.
echo.
echo   Installation directory: !INSTALL_DIR!
echo   CLI executable:        !INSTALL_DIR!\Scripts\rag.exe
echo   Config file:           %USERPROFILE%\.rag-kit.yaml
echo   Watch folder:          %WATCH_FOLDER%
echo   Vector DB:             %USERPROFILE%\lancedb
echo   Model cache:           %USERPROFILE%\models
echo.
echo   Quick start:
echo     Drop documents in: %WATCH_FOLDER%
echo     Then query:        rag query --json "your question"
echo     Check status:      rag status
echo.
echo   For help: rag --help
echo.

pause
exit /b 0

:end_fail
echo.
echo ============================================================
echo   Installation FAILED
echo ============================================================
echo   Please check the errors above and try again.
echo   For help: https://github.com/lychee888/Rag-Kit
echo.
pause
exit /b 1

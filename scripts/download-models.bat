@echo off
setlocal enabledelayedexpansion
title rag-kit Model Downloader (Windows)

REM ============================================================
REM   rag-kit - Model Downloader (Windows)
REM   Pre-downloads pinned models for fully offline rag-kit use.
REM   Prefers pinned GitHub Release tarballs (with integrity
REM   checks); falls back to the HuggingFace hub if unavailable.
REM
REM   Model weights are Apache-2.0, re-distributed with attribution -
REM   see THIRD_PARTY_NOTICES.md (and licenses/APACHE-2.0.txt).
REM ============================================================
echo.
echo ============================================================
echo   rag-kit - Model Downloader
echo   Prefers pinned GitHub Release tarballs (fast, version-locked)
echo ============================================================
echo.
set "MODEL_DIR=%USERPROFILE%\models"

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.10+ first.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Optionally set China mirror (for HuggingFace fallback below)
set "HF_MIRROR="
if not defined RAG_KIT_BATCH (
    set /p HF_CHOICE="Use China mirror hf-mirror.com for HF fallback? [y/N]: "
    if /i "!HF_CHOICE!"=="y" (
        set "HF_ENDPOINT=https://hf-mirror.com"
        echo Using mirror: !HF_ENDPOINT!
        echo.
    )
)

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

echo.
echo Models to download:
echo   1. paraphrase-multilingual-MiniLM-L12-v2 (embedding, ~470 MB)
echo   2. EasyOCR ch_sim + en (OCR, ~100 MB download)
echo   3. SmolVLM-256M-Instruct (VLM, ~500 MB) - optional
echo.
REM Only prompt if the caller didn't already specify (e.g. via installer).
if not defined SKIP_VLM (
    set /p DOWNLOAD_VLM="Download SmolVLM? [Y/n]: "
    if /i "!DOWNLOAD_VLM!"=="n" (
        set "SKIP_VLM=1"
    ) else (
        set "SKIP_VLM=0"
    )
) else (
    echo   VLM choice inherited from environment ^(SKIP_VLM=!SKIP_VLM!^)
)

echo.
echo Downloading models to: %MODEL_DIR%
echo This may take a while on first run...
echo.

REM ---- Python download script (real .py file, avoids fragile echo-block) ----
setlocal
if defined HF_ENDPOINT set "HF_ENDPOINT=%HF_ENDPOINT%"
set "MODEL_DIR=%MODEL_DIR%"
set "SKIP_VLM=%SKIP_VLM%"

REM Isolate from any inherited PYTHONPATH (see install-windows.bat).
set "PYTHONPATH=%TEMP%\ragkit_empty_pypath"

python "%~dp0_download_models.py"
set "DL_EXIT=%ERRORLEVEL%"

echo.
if %DL_EXIT% NEQ 0 (
    echo ERROR: Model download failed. Check the errors above.
    echo You may need to try again or use a VPN/proxy.
    pause
    exit /b 1
)

echo ============================================================
echo   Download Complete
echo ============================================================
echo Models cached in: %MODEL_DIR%
echo You can now install rag-kit offline:
echo   scripts\install-windows.bat
echo.
REM Pause only when run interactively (not when called from the installer).
if defined RAG_KIT_BATCH (
    exit /b 0
)
pause
exit /b 0

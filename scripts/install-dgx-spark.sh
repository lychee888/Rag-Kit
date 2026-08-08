#!/usr/bin/env bash
# install-dgx-spark.sh — One-shot installer for rag-kit on Linux (DGX Spark ARM64)
#
# Usage:
#   chmod +x install-dgx-spark.sh
#   ./install-dgx-spark.sh
#
# Tested on: NVIDIA DGX Spark (Grace ARM CPU + Blackwell GPU), Ubuntu 24.04

set -euo pipefail

# ---- Colors ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  rag-kit Installer - Linux (DGX Spark / ARM64 + CUDA)${NC}"
echo -e "${BLUE}  Agentic RAG System for Hermes Agent${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

INSTALL_DIR="$HOME/rag-kit-venv"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Step 1: Check Python >= 3.10 ----
echo -e "[1/8] Checking Python..."

PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYVER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
        MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON="$cmd"
            echo "  Found $cmd $PYVER"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "  ${RED}ERROR: Python 3.10+ not found.${NC}"
    echo "  Install with: sudo apt-get install -y python3 python3-venv python3-pip"
    exit 1
fi

echo "  OK: $PYTHON version $PYVER meets requirements"
echo ""

# ---- Step 2: Detect GPU (always CUDA on DGX, but verify) ----
echo -e "[2/8] Detecting GPU..."

USE_CUDA=0
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    USE_CUDA=1
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    echo "  GPU detected: $GPU_NAME (compute cap $GPU_CC)"
    echo "  Will install a PyTorch build compatible with this GPU."
else
    echo -e "  ${YELLOW}No NVIDIA GPU detected. Using CPU mode.${NC}"
    echo "  Note: DGX Spark should have a Blackwell GPU — check nvidia-smi if missing."
fi

# Detect ARM64
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "  Architecture: ARM64 (DGX Spark)"
else
    echo -e "  ${YELLOW}Architecture: $ARCH (not ARM64 — this script is optimized for DGX Spark)${NC}"
fi
echo ""

# ---- Step 2b: System dependencies ----
echo "  Checking system dependencies..."

if ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
    echo "  Installing python3-venv..."
    if command -v apt-get >/dev/null && command -v sudo >/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3-venv python3-pip 2>/dev/null || {
            echo -e "  ${YELLOW}Could not auto-install python3-venv (you may need a password).${NC}"
            echo "    sudo apt-get install -y python3-venv python3-pip"
        }
    else
        echo -e "  ${YELLOW}python venv module unavailable and no apt-get/sudo found.${NC}"
    fi
fi
echo ""

# ---- Step 3: Create virtual environment ----
echo -e "[3/8] Creating virtual environment at $INSTALL_DIR..."

if [ -d "$INSTALL_DIR" ]; then
    echo "  Removing existing installation..."
    rm -rf "$INSTALL_DIR"
fi

$PYTHON -m venv "$INSTALL_DIR"
if [ $? -ne 0 ]; then
    echo -e "  ${RED}ERROR: Failed to create virtual environment.${NC}"
    exit 1
fi
echo "  Created: $INSTALL_DIR"
echo ""

# ---- Step 4: Install dependencies ----
echo -e "[4/8] Installing dependencies..."

VENV_PYTHON="$INSTALL_DIR/bin/python"
VENV_PIP="$INSTALL_DIR/bin/pip"

# Upgrade pip; isolate from any inherited PYTHONPATH so pip only sees this venv.
unset PYTHONPATH 2>/dev/null || true
"$VENV_PYTHON" -m pip install --upgrade pip -q 2>/dev/null || true

TORCH_OK=0
if [ "$USE_CUDA" -eq 1 ]; then
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        echo "  Installing PyTorch with CUDA (aarch64)..."
        "$VENV_PIP" install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu126 -q 2>/dev/null && TORCH_OK=1
    else
        echo "  Installing PyTorch (CUDA-enabled build — cu128, supports Blackwell)..."
        # Explicit CUDA build: the default PyPI torch is CPU-only on some platforms.
        "$VENV_PIP" install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu128 -q 2>/dev/null && TORCH_OK=1
    fi
    if [ "$TORCH_OK" -eq 0 ]; then
        echo -e "  ${YELLOW}WARNING: cu128 build failed; trying cu124 (for older GPUs)...${NC}"
        "$VENV_PIP" install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu124 -q 2>/dev/null && TORCH_OK=1
    fi
    if [ "$TORCH_OK" -eq 0 ]; then
        echo -e "  ${YELLOW}WARNING: CUDA PyTorch failed; falling back to CPU PyTorch...${NC}"
        "$VENV_PIP" install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu -q && TORCH_OK=1
    fi
else
    echo "  Installing PyTorch (CPU)..."
    "$VENV_PIP" install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q && TORCH_OK=1
fi

if [ "$TORCH_OK" -ne 1 ]; then
    echo -e "  ${RED}ERROR: Failed to install PyTorch.${NC}"
    exit 1
fi

echo "  Installing rag-kit and dependencies..."
cd "$PROJECT_DIR"
"$VENV_PIP" install -e ".[ocr]" -q

if [ $? -ne 0 ]; then
    echo -e "  ${RED}ERROR: Failed to install rag-kit.${NC}"
    exit 1
fi
echo "  Installation complete."
echo ""

# ---- Step 5: Verify imports and CLI ----
echo -e "[5/8] Verifying installation..."

"$VENV_PYTHON" -c "import rag_kit; print('  rag_kit v' + rag_kit.__version__)" 2>/dev/null || {
    echo -e "  ${RED}ERROR: Failed to import rag_kit.${NC}"
    exit 1
}

"$VENV_PYTHON" -c "import rag_kit.autostart; print('  autostart OK')" 2>/dev/null
"$VENV_PYTHON" -c "import rag_kit.watcher; print('  watcher OK')" 2>/dev/null

# Heavy-stack import check (catches broken sentence-transformers / tokenizers)
if "$VENV_PYTHON" -c "import sentence_transformers, tokenizers; print('  ML stack OK (tokenizers ' + tokenizers.__version__ + ')')" 2>/dev/null; then
    echo "  ML stack: sentence-transformers OK"
else
    echo -e "  ${YELLOW}WARNING: sentence-transformers import check failed (dependency version conflict?).${NC}"
    echo "  Reinstall with:  pip install -e '.[ocr]'"
fi

# Test CLI
if "$VENV_PYTHON" -m rag_kit.cli.main --version &>/dev/null; then
    echo "  CLI: rag --version OK"
else
    echo -e "  ${YELLOW}WARNING: rag CLI test failed${NC}"
fi

# Create default config
"$VENV_PYTHON" -m rag_kit.cli.main config init &>/dev/null && \
    echo "  Config: Created default config (~/.rag-kit.yaml)" || \
    echo "  WARNING: Could not create default config (non-fatal)"

# Create default watch folder
mkdir -p "$HOME/Documents/rag-ingest" 2>/dev/null && \
    echo "  Created watch folder: ~/Documents/rag-ingest"

echo "  Verification passed."
echo ""

# ---- Step 6: Setup autostart ----
echo -e "[6/8] Setting up autostart..."

"$INSTALL_DIR/bin/rag" setup-autostart --json 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  Autostart: Installed (systemd user service + linger)"
else
    echo -e "  ${YELLOW}Autostart: NOT installed${NC}"
    echo "  To install autostart manually:"
    echo "    $INSTALL_DIR/bin/rag setup-autostart"
fi
echo ""

# ---- Step 7: Start watcher (single instance, lock-protected) ----
echo -e "[7/8] Starting watcher..."

# Prefer the systemd user service (the same one autostart installed) so
# only ONE watcher ever runs per folder — rag watch holds a per-folder
# lock, so even if the installer start and the boot-time autostart race,
# neither can double-ingest.
WATCH_FOLDER="$HOME/Documents/rag-ingest"

# Backfill existing files first: "rag watch" reacts only to NEW events,
# so pre-existing documents need a one-time ingest to be indexable.
echo "  Backfilling existing documents..."
if "$INSTALL_DIR/bin/rag" ingest "$WATCH_FOLDER" --json >/dev/null 2>&1; then
    if "$INSTALL_DIR/bin/rag" list-files --json 2>/dev/null | grep -q "file_count"; then
        echo "  Backfill: complete (documents indexed)"
    else
        echo "  Backfill: nothing to ingest yet (folder is empty; new files are picked up live)"
    fi
else
    echo "  Backfill: nothing to ingest yet"
fi

if systemctl --user is-enabled rag-kit-watcher &>/dev/null 2>&1; then
    systemctl --user start rag-kit-watcher 2>/dev/null || true
    sleep 2
    if systemctl --user is-active rag-kit-watcher &>/dev/null; then
        echo "  Watcher: Running (systemd user service)"
        echo "  Watch folder: $WATCH_FOLDER"
        echo "  Log: journalctl --user -u rag-kit-watcher"
    else
        echo -e "  ${YELLOW}Watcher: systemd service present but not active. Starting temporarily...${NC}"
        nohup "$INSTALL_DIR/bin/rag" watch "$WATCH_FOLDER" \
            > "$HOME/.rag-kit-watcher.log" 2>&1 &
        echo "  Watcher: Running (background, log: ~/.rag-kit-watcher.log)"
    fi
else
    nohup "$INSTALL_DIR/bin/rag" watch "$WATCH_FOLDER" \
        > "$HOME/.rag-kit-watcher.log" 2>&1 &
    echo "  Watcher: Running (background, log: ~/.rag-kit-watcher.log)"
fi
echo ""

# ---- Step 8 (optional): Pre-download pinned models ----
echo ""
echo "============================================================"
echo "  Optional: pre-download models for fully offline use."
echo "  (embedding ~470 MB + EasyOCR ~100 MB; VLM ~330 MB optional)"
echo "============================================================"
DL_MODELS=""
if [ -t 0 ]; then
    read -r -p "Pre-download pinned models now (from GitHub Releases)? [y/N]: " DL_MODELS || true
fi
if [ "$DL_MODELS" = "y" ] || [ "$DL_MODELS" = "Y" ]; then
    # Include SmolVLM so VLM chart/image captioning works out of the box.
    MODEL_DIR="$HOME/models" SKIP_VLM=0 bash "$SCRIPT_DIR/download-models.sh" || true
fi

SKILL_SRC="$PROJECT_DIR/SKILL.md"
SKILL_DST="$HOME/.hermes/skills/research/rag-kit/SKILL.md"

if [ -f "$SKILL_SRC" ]; then
    if [ ! -f "$SKILL_DST" ]; then
        mkdir -p "$(dirname "$SKILL_DST")" 2>/dev/null
        cp "$SKILL_SRC" "$SKILL_DST" 2>/dev/null && \
            echo "  Hermes Agent skill installed: research/rag-kit" || true
    fi
fi

# ---- Summary ----
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Installation Complete${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# Check autostart status
if systemctl --user is-active rag-kit-watcher &>/dev/null; then
    echo "  Autostart:    Installed (systemd user service)"
else
    echo "  Autostart:    NOT installed"
fi

# Check if watcher is running (systemd service, else report status)
if systemctl --user is-active rag-kit-watcher &>/dev/null; then
    echo "  Watcher:      Running (systemd user service)"
else
    if pgrep -f "rag watch" >/dev/null 2>&1; then
        echo "  Watcher:      Running (background)"
    else
        echo "  Watcher:      Not running (start with: $INSTALL_DIR/bin/rag watch $WATCH_FOLDER &)"
    fi
fi

echo ""
echo "  Installation directory: $INSTALL_DIR"
echo "  CLI executable:         $INSTALL_DIR/bin/rag"
echo "  Config file:            ~/.rag-kit.yaml"
echo "  Watch folder:           $WATCH_FOLDER"
echo "  Vector DB:              ~/lancedb"
echo "  Model cache:            ~/models"
echo ""
echo "  Quick start:"
echo "    Drop documents in: $WATCH_FOLDER"
echo "    Then query:        $INSTALL_DIR/bin/rag query --json \"your question\""
echo "    Check status:      $INSTALL_DIR/bin/rag status"
echo "    Add to PATH:       echo 'export PATH=\"$INSTALL_DIR/bin:\$PATH\"' >> ~/.bashrc"
echo ""
echo "  For help: $INSTALL_DIR/bin/rag --help"
echo ""

read -r -p "Press Enter to finish..." _dummy 2>/dev/null || true
exit 0
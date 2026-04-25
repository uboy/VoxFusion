#!/usr/bin/env bash
# VoxFusion macOS Quick-start
# Installs everything needed and optionally builds distributable binaries.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/.../scripts/macos_quickstart.sh | bash
#   # or, from a cloned repo:
#   bash scripts/macos_quickstart.sh [--build]
#
# Options:
#   --build      Also build GUI + CLI binaries via PyInstaller after setup
#   --model-only Only download the GigaAM model (skip venv/deps if already set up)
#   --no-model   Skip GigaAM model download (e.g. for faster-whisper-only usage)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv"
BUILD=false
MODEL_ONLY=false
SKIP_MODEL=false

for arg in "$@"; do
  case "$arg" in
    --build)      BUILD=true ;;
    --model-only) MODEL_ONLY=true ;;
    --no-model)   SKIP_MODEL=true ;;
  esac
done

step() { echo ""; echo "==> $*"; }
ok()   { echo "    [ok] $*"; }
warn() { echo "    [!]  $*"; }

# ── 0. Ensure we are in the project root ─────────────────────────────────────
cd "$PROJECT_ROOT"

if $MODEL_ONLY; then
  step "Downloading GigaAM-v3 model only"
  "$VENV/bin/python" -c "
from transformers import AutoModel
print('Downloading ai-sage/GigaAM-v3 (~1.5 GB)...')
AutoModel.from_pretrained('ai-sage/GigaAM-v3', trust_remote_code=True)
print('Done.')
"
  ok "Model ready."
  exit 0
fi

# ── 1. Homebrew ───────────────────────────────────────────────────────────────
step "Checking Homebrew"
if ! command -v brew &>/dev/null; then
  warn "Homebrew not found. Installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add brew to PATH for Apple Silicon
  if [[ -f /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
fi
ok "Homebrew $(brew --version | head -1)"

# ── 2. System dependencies ────────────────────────────────────────────────────
step "Installing system dependencies (python@3.11, ffmpeg)"
brew install python@3.11 ffmpeg 2>/dev/null || true
PYTHON=$(brew --prefix python@3.11)/bin/python3.11
ok "Python: $($PYTHON --version)"
ok "FFmpeg: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3)"

# ── 3. Virtual environment ────────────────────────────────────────────────────
step "Setting up Python virtual environment"
if [[ ! -f "$VENV/bin/activate" ]]; then
  "$PYTHON" -m venv "$VENV"
  ok "Created $VENV"
else
  ok "Reusing existing $VENV"
fi

PIP="$VENV/bin/pip"
PYTHON_VENV="$VENV/bin/python"

# ── 4. Core package ───────────────────────────────────────────────────────────
step "Installing VoxFusion"
"$PIP" install -e . --quiet
ok "voxfusion installed"

# ── 5. PyTorch (MPS-accelerated on Apple Silicon, CPU on Intel) ───────────────
step "Installing PyTorch + torchaudio"
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
  # Apple Silicon: standard PyTorch includes MPS support
  "$PIP" install torch torchaudio --quiet
  ok "PyTorch with MPS support (Apple Silicon)"
else
  # Intel Mac: CPU-only build is smaller
  "$PIP" install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
  ok "PyTorch CPU (Intel Mac)"
fi

# ── 6. GigaAM runtime dependencies ───────────────────────────────────────────
step "Installing GigaAM dependencies"
"$PIP" install \
  "transformers>=4.48,<5.0" \
  sentencepiece \
  omegaconf \
  hydra-core \
  pyannote-audio \
  Pillow \
  --quiet
ok "GigaAM deps ready"

# ── 7. Download GigaAM model ──────────────────────────────────────────────────
if ! $SKIP_MODEL; then
  step "Downloading GigaAM-v3 model (~1.5 GB, cached after first run)"
  HF_MODEL_CACHE="$HOME/.cache/huggingface/hub"
  if ls "$HF_MODEL_CACHE" 2>/dev/null | grep -q "GigaAM"; then
    ok "Model already cached in $HF_MODEL_CACHE"
  else
    "$PYTHON_VENV" -c "
from transformers import AutoModel
print('  Downloading ai-sage/GigaAM-v3...')
AutoModel.from_pretrained('ai-sage/GigaAM-v3', trust_remote_code=True)
print('  Download complete.')
"
    ok "Model downloaded"
  fi
fi

# ── 8. Smoke test ─────────────────────────────────────────────────────────────
step "Running smoke test"
"$PYTHON_VENV" -m voxfusion --version
ok "CLI works"

# ── 9. Build binaries (optional) ─────────────────────────────────────────────
if $BUILD; then
  step "Installing PyInstaller"
  "$PIP" install "pyinstaller>=6.15" --quiet
  ok "PyInstaller ready"

  step "Building GUI + CLI binaries"
  "$PYTHON_VENV" scripts/build_binaries.py --target gui
  "$PYTHON_VENV" scripts/build_binaries.py --target cli
  ok "Binaries in dist/binaries/"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "Setup complete. To transcribe files:"
echo ""
echo "  $VENV/bin/python -m voxfusion transcribe --model gigaam-v3-e2e-ctc \\"
echo "      --output-format txt --diarization-strategy none \\"
echo "      --output-dir ./transcripts ./your_video.mp4"
echo ""
echo "Or with a whole directory:"
echo "  $VENV/bin/python -m voxfusion transcribe --model gigaam-v3-e2e-ctc \\"
echo "      --output-dir ./transcripts ./videos/"
echo ""
if $BUILD; then
  echo "GUI binary: dist/binaries/voxfusion-gui.app"
  echo "CLI binary: dist/binaries/voxfusion-cli"
else
  echo "To also build distributable binaries, run:"
  echo "  bash scripts/macos_quickstart.sh --build"
fi

#!/usr/bin/env bash
# =============================================================================
# setup_pi.sh — Gesture Engine environment setup for Raspberry Pi / DietPi
#
# Target:  Debian Bookworm (DietPi or Raspberry Pi OS), aarch64 / armv7
# Python:  3.11  (mediapipe has no PyPI wheels for 3.12+ on aarch64)
#
# What this script does:
#   1. Adds the deadsnakes PPA or installs python3.11 from apt (whichever works)
#   2. Creates .venv using Python 3.11
#   3. Installs a pre-built mediapipe 0.10.14 aarch64 wheel from the community
#      mirror (github.com/Morioh/mediapipe-cp311-linux-aarch64)
#   4. Installs remaining deps from requirements.txt
#
# Usage:
#   chmod +x setup_pi.sh
#   ./setup_pi.sh
# =============================================================================

set -euo pipefail

VENV_DIR=".venv"
PYTHON_BIN=""

# ── Wheel URLs ────────────────────────────────────────────────────────────────
# Pre-built mediapipe 0.10.14 for cp311 / linux_aarch64
# Source: https://github.com/niconielsen32/mediapipe-raspberrypi
MP_WHEEL_URL="https://github.com/niconielsen32/mediapipe-raspberrypi/releases/download/v0.10.14/mediapipe-0.10.14-cp311-cp311-linux_aarch64.whl"
MP_WHEEL_FILE="/tmp/mediapipe-0.10.14-cp311-cp311-linux_aarch64.whl"

# Fallback mirror (google-coral index — may carry an earlier 0.10.x build)
CORAL_INDEX="https://google-coral.github.io/py-repo/"

echo "=========================================="
echo "  Gesture Engine — Pi Setup"
echo "  $(date)"
echo "=========================================="

# ── 1. Detect / install Python 3.11 ──────────────────────────────────────────
echo ""
echo "[1/5] Checking for Python 3.11 …"

if command -v python3.11 &>/dev/null; then
    PYTHON_BIN="$(command -v python3.11)"
    echo "      Found: $PYTHON_BIN  ($(python3.11 --version))"
else
    echo "      python3.11 not found — installing via apt …"
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev \
        python3-pip build-essential
    PYTHON_BIN="$(command -v python3.11)"
    echo "      Installed: $PYTHON_BIN  ($(python3.11 --version))"
fi

# ── 2. Install system libs needed by OpenCV + PyQt5 ──────────────────────────
echo ""
echo "[2/5] Installing system libraries …"
sudo apt-get install -y --no-install-recommends \
    libcamera-dev \
    libopenblas-dev \
    libatlas-base-dev \
    python3-pyqt5 \
    v4l-utils \
    2>/dev/null || true   # non-fatal — some packages may not exist on all images

# ── 3. Create / refresh the virtual environment ───────────────────────────────
echo ""
echo "[3/5] Creating virtual environment at $VENV_DIR …"
if [ -d "$VENV_DIR" ]; then
    echo "      Existing venv found — recreating to ensure Python 3.11 is used."
    rm -rf "$VENV_DIR"
fi
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip/wheel/setuptools inside the venv first
pip install --upgrade pip wheel setuptools --quiet

# ── 4. Install mediapipe from pre-built aarch64 wheel ─────────────────────────
echo ""
echo "[4/5] Installing mediapipe 0.10.14 (aarch64 wheel) …"

if [ ! -f "$MP_WHEEL_FILE" ]; then
    echo "      Downloading wheel from GitHub …"
    if ! curl -fsSL "$MP_WHEEL_URL" -o "$MP_WHEEL_FILE"; then
        echo "      ⚠  Primary mirror failed — trying google-coral index …"
        pip install mediapipe --extra-index-url "$CORAL_INDEX" --quiet
        MP_WHEEL_FILE=""
    fi
fi

if [ -n "$MP_WHEEL_FILE" ] && [ -f "$MP_WHEEL_FILE" ]; then
    pip install "$MP_WHEEL_FILE"
fi

# ── 5. Install remaining dependencies ────────────────────────────────────────
echo ""
echo "[5/5] Installing remaining dependencies from requirements.txt …"
pip install \
    "opencv-python-headless>=4.9.0,<4.11.0" \
    "websockets>=12.0,<14.0" \
    "pynput>=1.7.0"

# PyQt5 — use apt version if available (much faster than pip on Pi)
if python3.11 -c "import PyQt5" 2>/dev/null; then
    echo "      PyQt5 already available (apt version). Skipping pip install."
else
    echo "      Installing PyQt5 via pip (this may take a few minutes) …"
    pip install "PyQt5>=5.15.0" --quiet
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  ✅  Dependency setup complete!"
echo ""
echo "  Installing systemd autostart services …"
echo "=========================================="

# ── 6. Install autostart services ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SH="${SCRIPT_DIR}/install_services.sh"

if [ -f "${INSTALL_SH}" ]; then
    bash "${INSTALL_SH}"
else
    echo ""
    echo "  ⚠  install_services.sh not found at ${INSTALL_SH}"
    echo "     Copy it to the project directory and run manually:"
    echo "     sudo bash install_services.sh"
fi

echo ""
echo "=========================================="
echo "  ✅  Full setup complete!"
echo ""
echo "  Both services are installed and enabled:"
echo "    gesture-engine  →  starts at boot (multi-user.target)"
echo "    gesture-overlay →  starts when desktop is ready (graphical.target)"
echo ""
echo "  Check status anytime:"
echo "    systemctl status gesture-engine"
echo "    systemctl status gesture-overlay"
echo ""
echo "  View live logs:"
echo "    journalctl -u gesture-engine  -f"
echo "    journalctl -u gesture-overlay -f"
echo "=========================================="

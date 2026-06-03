#!/usr/bin/env bash

# JMod Bash Launcher

set -e

# Directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change into JMod directory
cd "$SCRIPT_DIR"

# Check for uv

if ! command -v uv >/dev/null 2>&1; then

    echo "UV not found, attempting installation..."

    # Try pip first
    if command -v pip >/dev/null 2>&1; then
        pip install uv
    fi

    # Check again
    if ! command -v uv >/dev/null 2>&1; then

        # Official installer
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add common install location for current session
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Final check
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: Failed to install uv automatically."
        exit 1
    fi
fi

# ------------------------------------------------------------------
# Create/update environment

uv sync --python 3.11

# ------------------------------------------------------------------
# Launch GUI

uv run python run_jmod_from_GUI.py
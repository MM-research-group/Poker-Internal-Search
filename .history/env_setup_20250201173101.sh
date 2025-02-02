#!/usr/bin/env bash

# env_setup.sh
# This script creates and sets up the conda environment for the Poker codebase.
# It will automatically create the environment if it does not already exist,
# activate it, and install required dependencies.

# Check if conda is installed.
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

ENV_NAME="poker_env"
PYTHON_VERSION="3.9"
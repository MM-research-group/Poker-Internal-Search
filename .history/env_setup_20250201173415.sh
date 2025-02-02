#!/usr/bin/env bash

# env_setup.sh
# This script creates and sets up the conda environment for the Poker codebase.
# It will automatically create the environment if it does not already exist,
# activate it, and install required dependencies.


# chmod +x env_setup.sh


# Check if conda is installed.
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

ENV_NAME="poker_env"
PYTHON_VERSION="3.11"

# Check if the environment already exists.
if conda info --envs | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Conda environment '${ENV_NAME}' already exists."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -y --name "${ENV_NAME}" python=${PYTHON_VERSION}
fi

# Activate the environment.
echo "Activating conda environment '${ENV_NAME}'..."
# Source conda's shell script to enable 'conda activate'
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install -r "$SCRIPT_DIR/requirements.txt"
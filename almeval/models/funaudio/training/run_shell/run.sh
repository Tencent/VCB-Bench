#!/bin/bash

# ============================================================================
# Training Script for Fun-Audio-Chat
# ============================================================================
# This script trains the FunAudioChat model using LLaMA-Factory with multimodal
# audio capabilities through the funaudiochat and training.plugin modules.
#
# Requirements:
#   - LLaMA-Factory installed (see third_party/LLaMA-Factory)
#   - funaudiochat/ and training/plugin/ modules available
#   - Training configuration file (training/configs/sft.yaml)
#   - Dataset info (training/data/dataset_info.json)
#
# Usage:
#   Single GPU:     bash training/run_shell/run.sh
#   Multi-GPU:      Adjust NNODES and NODE_RANK as needed
#   Multi-node:     Set MASTER_ADDR, NNODES, NODE_RANK on each node
# ============================================================================

set -e  # Exit on error

export AUDIO_PLACEHOLDER="<|audio_bos|><|AUDIO|><|audio_eos|>"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ============================================================================
# Environment Variables
# ============================================================================

# SwanLab configuration (optional, for experiment tracking)
# export SWANLAB_API_KEY="your_api_key_here"
# export SWANLAB_PROJECT="Fun-Audio-Chat"

# Disable version check for faster startup
export DISABLE_VERSION_CHECK=1

# Force torchrun for distributed training
export FORCE_TORCHRUN=1

# Multi-node training configuration
# Number of nodes (default: 1 for single-node training)
export NNODES="${NNODES:-1}"

# Node rank (0-indexed, set by your job scheduler in multi-node setup)
export NODE_RANK="${NODE_RANK:-${RANK:-0}}"

# Master address for multi-node training (required when NNODES > 1)
# export MASTER_ADDR="10.0.0.1"

# Add project root and plugin dir to Python path
# Plugin dir contains sitecustomize.py which auto-registers plugins
PLUGIN_DIR="${PROJECT_ROOT}/training/plugin"
export PYTHONPATH="${PLUGIN_DIR}:${PROJECT_ROOT}:${PYTHONPATH}"

# DDP timeout (increase for slow networks or large models)
export TIMEOUT=180000000

# ============================================================================
# Configuration
# ============================================================================

# Path to training configuration file
CONFIG_FILE="${CONFIG_FILE:-configs/sft.yaml}"

# Log directory
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Extract experiment name from config file
EXPERIMENT_NAME=$(basename "${CONFIG_FILE}" .yaml)

# Log file path
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}_RANK${NODE_RANK}.log"

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo "========================================"
echo "Fun-Audio-Chat Training Script"
echo "========================================"
echo "Configuration:"
echo "  - Config file: ${CONFIG_FILE}"
echo "  - Number of nodes: ${NNODES}"
echo "  - Node rank: ${NODE_RANK}"
echo "  - Master address: ${MASTER_ADDR:-localhost}"
echo "  - Log file: ${LOG_FILE}"
echo "========================================"

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Configuration file not found: ${CONFIG_FILE}"
    echo "Please create a training configuration file or specify CONFIG_FILE environment variable."
    exit 1
fi

# Check if LLaMA-Factory is available
if ! command -v llamafactory-cli &> /dev/null; then
    echo "Error: llamafactory-cli not found in PATH"
    echo "Please install LLaMA-Factory:"
    echo "  cd third_party/LLaMA-Factory && pip install -e ."
    exit 1
fi

echo "Starting training..."
echo ""

# ============================================================================
# Training
# ============================================================================

# Resume training (optional)
# Uncomment and set these variables to resume from a checkpoint:
# export SWANLAB_RESUME=must
# export SWANLAB_RUN_ID="your_run_id_here"
# Or modify config file to include: resume_from_checkpoint: true

# Clean up any previous training processes (use with caution!)
# pkill -f python

# Start training with logging
echo "Starting training at $(date)"
echo "Config: ${CONFIG_FILE}"
echo "Logs will be saved to: ${LOG_FILE}"
echo ""

# Use llamafactory-cli with sitecustomize.py for auto-registration
# The sitecustomize.py in training/plugin/ (added to PYTHONPATH) ensures
# plugins are registered in every Python process (including torchrun workers)
llamafactory-cli train "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

# ============================================================================
# Post-training
# ============================================================================

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

if [ ${TRAINING_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Training completed successfully!"
    echo "Logs saved to: ${LOG_FILE}"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Training failed with exit code: ${TRAINING_EXIT_CODE}"
    echo "Please check logs: ${LOG_FILE}"
    echo "========================================"
    exit ${TRAINING_EXIT_CODE}
fi


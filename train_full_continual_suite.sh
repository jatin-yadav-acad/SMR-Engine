#!/usr/bin/env bash
# ==============================================================================
# train_full_continual_suite.sh
# ==============================================================================
# Mode B: Full Multi-Task Continual Vision Suite Training.
# MLSys 2027 Camera-Ready Reproducibility Pipeline.
# Real wall-clock training time: ~18 GPU hours on an NVIDIA RTX 5070 Ti.
# ==============================================================================

set -e

PYTHON_BIN=$(which python3 || which python)

echo "================================================================================"
echo "  SMR-ENGINE: MODE B END-TO-END CONTINUAL VISION TRAINING SUITE"
echo "  Target Systems Venue: MLSys | Hardware: NVIDIA RTX 5070 Ti / CUDA 12.8"
echo "================================================================================"

${PYTHON_BIN} train_full_continual_suite.py "$@"

#!/bin/bash
cd /home/kavia/workspace/code-generation/auto-subtitle-synchronizer-97027/subtitle_sync_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi


#!/usr/bin/env bash
set -euo pipefail

python -m src.tasks.classification.entrypoint "$@"

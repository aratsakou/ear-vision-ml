#!/usr/bin/env bash
set -euo pipefail

python -m src.core.data.dataset_builder "$@"

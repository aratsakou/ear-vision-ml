#!/usr/bin/env bash
set -euo pipefail

python -m src.core.export.exporter "$@"

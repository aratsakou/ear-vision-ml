#!/usr/bin/env bash
set -euo pipefail

python -m src.runtimes.video_inference.offline_runner "$@"

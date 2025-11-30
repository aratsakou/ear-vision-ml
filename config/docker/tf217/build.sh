#!/usr/bin/env bash
set -euo pipefail

IMAGE_URI="${1:-ear-vision-ml:tf217}"
docker build -f config/docker/tf217/Dockerfile -t "${IMAGE_URI}" .
echo "Built: ${IMAGE_URI}"

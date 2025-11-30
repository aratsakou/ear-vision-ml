#!/usr/bin/env bash
set -euo pipefail

IMAGE_URI="${1:?Provide image URI, e.g. europe-docker.pkg.dev/<proj>/<repo>/ear-vision-ml:tf217}"
docker push "${IMAGE_URI}"
echo "Pushed: ${IMAGE_URI}"

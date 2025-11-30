#!/bin/bash
set -e

# Usage: ./scripts/vertex_submit.sh <TASK> <CONFIG_NAME> <GCS_STAGING> <REGION> [IMAGE_URI]
# Example: ./scripts/vertex_submit.sh classification config gs://my-bucket/staging europe-west2

TASK=$1
CONFIG_NAME=$2
STAGING_BUCKET=$3
REGION=${4:-europe-west2}
IMAGE_URI=${5:-europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-17.py310:latest}

if [ -z "$TASK" ] || [ -z "$CONFIG_NAME" ] || [ -z "$STAGING_BUCKET" ]; then
    echo "Usage: ./scripts/vertex_submit.sh <TASK> <CONFIG_NAME> <GCS_STAGING> <REGION> [IMAGE_URI]"
    exit 1
fi

JOB_NAME="${TASK}_$(date +%Y%m%d_%H%M%S)"

echo "Submitting job $JOB_NAME to Vertex AI ($REGION)..."

gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=$JOB_NAME \
  --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=$IMAGE_URI \
  --args="src/tasks/${TASK}/entrypoint.py","task=${TASK}","config_name=${CONFIG_NAME}","run.log_vertex_experiments=true"

echo "Job submitted. Check Vertex AI console."

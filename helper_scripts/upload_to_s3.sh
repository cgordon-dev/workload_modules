#!/bin/bash
# S3 Upload Script for DCGAN Results
# This script uploads DCGAN training logs, models, and results to an S3 bucket

# Configuration
S3_BUCKET="aws-gpu-monitoring-logs"
S3_PREFIX="dcgan-results/$(date +%Y%m%d)"
PROJECT_DIR="/home/ubuntu/gpu-monitoring/gpu_monitoring_project/dcgan_model_v2"
LOG_DIR="${PROJECT_DIR}/logs"
MODEL_DIR="${PROJECT_DIR}/models"
SAMPLES_DIR="${PROJECT_DIR}/samples"
OUTPUT_DIR="${PROJECT_DIR}/output"

# Create a timestamp for this upload
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
UPLOAD_LOG="${LOG_DIR}/s3_upload_${TIMESTAMP}.log"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${UPLOAD_LOG}"
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    log "Error: AWS CLI not found. Please install it with 'pip install awscli'"
    exit 1
fi

# Check if we have AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    log "Error: AWS credentials not found or not valid"
    exit 1
fi

# Create a metadata file with instance information
METADATA_FILE="/tmp/dcgan_metadata_${TIMESTAMP}.json"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null | sed 's/[a-z]$//' || echo "unknown")

cat > "${METADATA_FILE}" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "instance_id": "${INSTANCE_ID}",
    "instance_type": "${INSTANCE_TYPE}",
    "region": "${REGION}",
    "upload_batch": "${TIMESTAMP}"
}
EOF

log "Starting upload to S3 bucket: ${S3_BUCKET}/${S3_PREFIX}"
log "Instance metadata: ID=${INSTANCE_ID}, Type=${INSTANCE_TYPE}, Region=${REGION}"

# Upload metadata
log "Uploading instance metadata"
aws s3 cp "${METADATA_FILE}" "s3://${S3_BUCKET}/${S3_PREFIX}/metadata/${TIMESTAMP}_metadata.json"

# Upload logs
if [ -d "${LOG_DIR}" ]; then
    log "Uploading logs from ${LOG_DIR}"
    aws s3 sync "${LOG_DIR}" "s3://${S3_BUCKET}/${S3_PREFIX}/logs/" \
        --exclude "*" \
        --include "*.log" \
        --include "*.txt"
fi

# Upload models
if [ -d "${MODEL_DIR}" ]; then
    log "Uploading models from ${MODEL_DIR}"
    aws s3 sync "${MODEL_DIR}" "s3://${S3_BUCKET}/${S3_PREFIX}/models/" \
        --exclude "*" \
        --include "*.pth" \
        --include "final_model.pth"
fi

# Upload sample images
if [ -d "${SAMPLES_DIR}" ]; then
    log "Uploading sample images from ${SAMPLES_DIR}"
    aws s3 sync "${SAMPLES_DIR}" "s3://${S3_BUCKET}/${S3_PREFIX}/samples/" \
        --exclude "*" \
        --include "*.png"
fi

# Upload output results
if [ -d "${OUTPUT_DIR}" ]; then
    log "Uploading evaluation results from ${OUTPUT_DIR}"
    aws s3 sync "${OUTPUT_DIR}" "s3://${S3_BUCKET}/${S3_PREFIX}/results/" \
        --exclude "*" \
        --include "*.png" \
        --include "*.txt" \
        --include "evaluation_results.txt"
fi

# Upload this log file itself
log "Upload completed successfully"
aws s3 cp "${UPLOAD_LOG}" "s3://${S3_BUCKET}/${S3_PREFIX}/logs/$(basename ${UPLOAD_LOG})"

echo "All DCGAN data uploaded to s3://${S3_BUCKET}/${S3_PREFIX}/"
#!/bin/bash
# Simple StyleGAN2 Training with S3 Upload
# This script runs your StyleGAN workflow and uploads results to S3

# Directory paths
PROJECT_DIR="$(pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
STYLEGAN_SCRIPT="${PROJECT_DIR}/run_stylegan2.sh"

# S3 configuration
S3_BUCKET="aws-gpu-monitoring-logs"
S3_PREFIX="stylegan2-results"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_DIR}/stylegan-run.log"
}

log "Starting StyleGAN2 workflow"

# Run the StyleGAN2 script
log "Running StyleGAN2 workflow"
bash "${STYLEGAN_SCRIPT}"
STYLEGAN_EXIT_CODE=$?

if [ $STYLEGAN_EXIT_CODE -ne 0 ]; then
    log "Error: StyleGAN2 workflow failed with exit code $STYLEGAN_EXIT_CODE"
else
    log "StyleGAN2 workflow completed successfully"
fi

# Upload to S3
log "Uploading results to S3..."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
S3_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/${TIMESTAMP}"

# Create metadata file
METADATA_FILE="${LOG_DIR}/metadata_${TIMESTAMP}.json"
cat > "${METADATA_FILE}" << EOL
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "exit_code": ${STYLEGAN_EXIT_CODE}
}
EOL

# Upload logs
aws s3 cp "${LOG_DIR}" "${S3_PATH}/logs/" --recursive --exclude "*" --include "*.log"

# Upload output directory
if [ -d "${PROJECT_DIR}/output" ]; then
    aws s3 cp "${PROJECT_DIR}/output" "${S3_PATH}/output/" --recursive
    log "Uploaded output directory to ${S3_PATH}/output/"
fi

# Upload models directory
if [ -d "${PROJECT_DIR}/models" ]; then
    aws s3 cp "${PROJECT_DIR}/models" "${S3_PATH}/models/" --recursive
    log "Uploaded models directory to ${S3_PATH}/models/"
fi

log "S3 upload completed: ${S3_PATH}"
log "StyleGAN2 run completed"

exit $STYLEGAN_EXIT_CODE

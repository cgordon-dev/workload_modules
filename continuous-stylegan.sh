#!/bin/bash
# Continuous StyleGAN2 Training with Automatic S3 Uploads
# This script sets up continuous operation of StyleGAN2 workflows
# with monitoring and automatic uploads to S3 upon completion

# Directory paths
PROJECT_DIR="$(pwd)"
DOCKER_COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yaml"
STYLEGAN_SCRIPT="${PROJECT_DIR}/stylegan-script.sh"
LOG_DIR="${PROJECT_DIR}/logs"
MONITORING_LOG="${LOG_DIR}/monitoring.log"
RUN_MARKER="${PROJECT_DIR}/.stylegan_running"
UPLOAD_MARKER="${PROJECT_DIR}/.upload_required"

# S3 configuration
S3_BUCKET="aws-gpu-monitoring-logs"
S3_PREFIX="stylegan2-results"

# Create necessary directories
mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_DIR}/continuous_runs"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MONITORING_LOG}"
}

log "Starting continuous StyleGAN2 operation"

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log "Docker daemon is not running. Attempting to start..."
        sudo systemctl start docker || true
        sleep 5
        if ! docker info > /dev/null 2>&1; then
            log "Failed to start Docker daemon. Exiting."
            return 1
        fi
    fi
    return 0
}

# Function to check and start Docker Compose stack
start_monitoring() {
    log "Checking Docker Compose monitoring stack"
    
    if [ ! -f "${DOCKER_COMPOSE_FILE}" ]; then
        log "Docker Compose file not found at ${DOCKER_COMPOSE_FILE}"
        return 1
    fi
    
    # Check if stack is already running
    if docker ps | grep -q "prometheus"; then
        log "Monitoring stack already running"
    else
        log "Starting monitoring stack"
        docker-compose -f "${DOCKER_COMPOSE_FILE}" up -d
        
        # Wait and verify all services are running
        sleep 10
        if [ $(docker-compose -f "${DOCKER_COMPOSE_FILE}" ps -q | wc -l) -ne $(docker-compose -f "${DOCKER_COMPOSE_FILE}" config --services | wc -l) ]; then
            log "Warning: Not all containers are running"
            docker-compose -f "${DOCKER_COMPOSE_FILE}" ps | tee -a "${MONITORING_LOG}"
            return 1
        fi
        log "Monitoring stack started successfully"
    fi
    
    return 0
}

# Function to upload data to S3
upload_to_s3() {
    local run_id=$1
    local status=$2
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    log "Uploading run $run_id data to S3 (status: $status)"
    
    # Create paths
    local S3_DATE_PREFIX="${S3_PREFIX}/$(date +%Y%m%d)/${timestamp}/run_${run_id}"
    local S3_LOG_PREFIX="${S3_DATE_PREFIX}/logs"
    local S3_MODEL_PREFIX="${S3_DATE_PREFIX}/models"
    local S3_IMAGE_PREFIX="${S3_DATE_PREFIX}/images"
    
    # Create metadata file
    local METADATA_FILE="${LOG_DIR}/metadata_${timestamp}.json"
    local INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
    local INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
    local REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null | sed 's/[a-z]$//' || echo "unknown")
    
    cat > "${METADATA_FILE}" << EOL
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "instance_id": "${INSTANCE_ID}",
    "instance_type": "${INSTANCE_TYPE}",
    "region": "${REGION}",
    "run_id": "${run_id}",
    "status": "${status}"
}
EOL
    
    # Upload metadata
    aws s3 cp "${METADATA_FILE}" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/metadata.json" || \
    log "Failed to upload metadata"
    
    # Upload logs
    log "Uploading log files"
    find "${LOG_DIR}" -name "*.log" -type f -mtime -1 | while read log_file; do
        aws s3 cp "${log_file}" "s3://${S3_BUCKET}/${S3_LOG_PREFIX}/$(basename ${log_file})" || \
        log "Failed to upload log file: ${log_file}"
    done
    
    # Find and upload master summary
    if [ -f "${PROJECT_DIR}/output/master_summary.txt" ]; then
        aws s3 cp "${PROJECT_DIR}/output/master_summary.txt" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/master_summary.txt" || \
        log "Failed to upload master summary"
    fi
    
    # Find and upload master gallery
    if [ -f "${PROJECT_DIR}/output/master_gallery.html" ]; then
        aws s3 cp "${PROJECT_DIR}/output/master_gallery.html" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/master_gallery.html" || \
        log "Failed to upload master gallery"
    fi
    
    # Upload models (only the final models for each workflow to save space)
    log "Uploading model files"
    find "${PROJECT_DIR}/output" -name "*final*.pkl" -type f | while read model_file; do
        aws s3 cp "${model_file}" "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/$(basename ${model_file})" || \
        log "Failed to upload model file: ${model_file}"
    done
    
    # Upload sample images (a selection to save space)
    log "Uploading sample images"
    for workflow in "basic_training" "finetuned" "latent_optimized" "mixed_precision"; do
        if [ -d "${PROJECT_DIR}/output/${workflow}/generated_images" ]; then
            # Create a directory for this workflow
            mkdir -p "${PROJECT_DIR}/continuous_runs/run_${run_id}/${workflow}"
            
            # Find and copy a sample of images (max 20 per workflow)
            find "${PROJECT_DIR}/output/${workflow}/generated_images" -name "*.png" -type f | head -20 | while read img_file; do
                local target_dir="${PROJECT_DIR}/continuous_runs/run_${run_id}/${workflow}"
                cp "${img_file}" "${target_dir}/"
                
                # Upload to S3
                aws s3 cp "${img_file}" "s3://${S3_BUCKET}/${S3_IMAGE_PREFIX}/${workflow}/$(basename ${img_file})" || \
                log "Failed to upload image: ${img_file}"
            done
        fi
    done
    
    log "S3 upload for run $run_id completed"
    return 0
}

# Function to run a complete StyleGAN2 workflow cycle
run_stylegan_cycle() {
    local run_id=$1
    local run_log="${LOG_DIR}/stylegan_run_${run_id}.log"
    
    log "Starting StyleGAN2 workflow cycle (Run ID: $run_id)"
    touch "${RUN_MARKER}"
    
    # Run the StyleGAN2 script
    bash "${STYLEGAN_SCRIPT}" > "${run_log}" 2>&1
    local exit_code=$?
    
    # Set status based on exit code
    local status="completed"
    if [ ${exit_code} -ne 0 ]; then
        status="failed"
        log "StyleGAN2 workflow failed with exit code ${exit_code}"
    else
        log "StyleGAN2 workflow completed successfully"
    fi
    
    # Create upload marker
    echo "${run_id}|${status}" > "${UPLOAD_MARKER}"
    
    # Remove run marker
    rm -f "${RUN_MARKER}"
    
    return ${exit_code}
}

# Function to check if an upload is required and perform it
check_and_upload() {
    if [ -f "${UPLOAD_MARKER}" ]; then
        local marker_content=$(cat "${UPLOAD_MARKER}")
        local run_id=$(echo "${marker_content}" | cut -d'|' -f1)
        local status=$(echo "${marker_content}" | cut -d'|' -f2)
        
        log "Found upload marker for run ${run_id} (status: ${status})"
        upload_to_s3 "${run_id}" "${status}"
        
        # Remove marker after successful upload
        rm -f "${UPLOAD_MARKER}"
    fi
}

# Function to clean up old data to prevent disk filling
cleanup_old_data() {
    log "Cleaning up old data"
    
    # Remove logs older than 7 days
    find "${LOG_DIR}" -name "*.log" -type f -mtime +7 -delete
    
    # Remove old output directories if disk space is low
    local disk_usage=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ ${disk_usage} -gt 85 ]; then
        log "Disk usage is high (${disk_usage}%), cleaning up old output directories"
        
        # Find and remove oldest continuous run directories
        find "${PROJECT_DIR}/continuous_runs" -maxdepth 1 -name "run_*" -type d | sort | head -n -5 | xargs rm -rf
        
        # Remove old model files except the most recent ones
        find "${PROJECT_DIR}/output" -name "*.pkl" -type f -not -name "*final*.pkl" -mtime +3 -delete
    fi
}

# Main continuous operation loop
run_id=1

# Check if there's an existing run counter
if [ -f "${PROJECT_DIR}/.run_counter" ]; then
    run_id=$(cat "${PROJECT_DIR}/.run_counter")
fi

# Main loop for continuous operation
while true; do
    # Check if Docker is running
    if ! check_docker; then
        log "Docker not available. Waiting 30 minutes before retry..."
        sleep 1800
        continue
    fi
    
    # Start monitoring stack if not running
    if ! start_monitoring; then
        log "Failed to start monitoring stack. Waiting 15 minutes before retry..."
        sleep 900
        continue
    fi
    
    # Check if there's a pending upload
    check_and_upload
    
    # Check if a run is already in progress
    if [ -f "${RUN_MARKER}" ]; then
        log "A StyleGAN2 run is already in progress. Waiting..."
        sleep 300
        continue
    fi
    
    # Clean up old data
    cleanup_old_data
    
    # Run a complete StyleGAN2 cycle
    log "Starting StyleGAN2 run #${run_id}"
    run_stylegan_cycle "${run_id}"
    
    # Save the next run ID
    run_id=$((run_id + 1))
    echo "${run_id}" > "${PROJECT_DIR}/.run_counter"
    
    # Wait for upload to complete
    check_and_upload
    
    # Wait before starting the next cycle
    log "Waiting 1 hour before starting the next cycle"
    sleep 3600
done

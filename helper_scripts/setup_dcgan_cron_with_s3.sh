#!/bin/bash
# DCGAN Cron Job Setup Script with S3 Upload
# This script sets up a cron job to run DCGAN training and inference on a schedule
# and upload results to an AWS S3 bucket

# Directory paths (update these to match your environment)
PROJECT_DIR="/home/ubuntu/gpu-monitoring/gpu_monitoring_project/dcgan_model_v2"
TRAIN_SCRIPT="${PROJECT_DIR}/dcgan_train.py"
INFERENCE_SCRIPT="${PROJECT_DIR}/fixed_dcgan_inference.py"
LOG_DIR="${PROJECT_DIR}/logs"
CHECKPOINT_DIR="${PROJECT_DIR}/models"
OUTPUT_DIR="${PROJECT_DIR}/output"
SAMPLES_DIR="${PROJECT_DIR}/samples"
VENV_PATH="/home/ubuntu/gpu-monitoring/venv/bin/activate"

# S3 configuration
S3_BUCKET="aws-gpu-monitoring-logs"
S3_PREFIX="dcgan-results"

# Create log and output directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Create the wrapper script that will be executed by cron
WRAPPER_SCRIPT="${PROJECT_DIR}/run_dcgan_scheduled.sh"
cat > "${WRAPPER_SCRIPT}" << 'EOF'
#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration (these will be replaced by the setup script)
TRAIN_SCRIPT="__TRAIN_SCRIPT__"
INFERENCE_SCRIPT="__INFERENCE_SCRIPT__"
LOG_DIR="__LOG_DIR__"
CHECKPOINT_DIR="__CHECKPOINT_DIR__"
OUTPUT_DIR="__OUTPUT_DIR__"
SAMPLES_DIR="__SAMPLES_DIR__"
VENV_PATH="__VENV_PATH__"
S3_BUCKET="__S3_BUCKET__"
S3_PREFIX="__S3_PREFIX__"

# Create timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRAIN_LOG="${LOG_DIR}/train_${TIMESTAMP}.log"
INFERENCE_LOG="${LOG_DIR}/inference_${TIMESTAMP}.log"
UPLOAD_LOG="${LOG_DIR}/s3_upload_${TIMESTAMP}.log"
FINAL_MODEL="${CHECKPOINT_DIR}/final_model.pth"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_DIR}/cron_job.log"
}

# Activate virtual environment if it exists
if [ -f "${VENV_PATH}" ]; then
    log "Activating Python virtual environment"
    source "${VENV_PATH}"
else
    log "Warning: Virtual environment not found at ${VENV_PATH}"
fi

# Navigate to project directory
cd "${SCRIPT_DIR}"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    log "Error: NVIDIA drivers not found. GPU may not be available."
    exit 1
fi

# Run training script
log "Starting DCGAN training..."
python3 "${TRAIN_SCRIPT}" > "${TRAIN_LOG}" 2>&1
TRAIN_EXIT_CODE=$?

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    log "Error: Training failed with exit code ${TRAIN_EXIT_CODE}. See log: ${TRAIN_LOG}"
    exit ${TRAIN_EXIT_CODE}
fi

log "Training completed successfully"

# Check if final model exists
if [ ! -f "${FINAL_MODEL}" ]; then
    log "Error: Final model not found at ${FINAL_MODEL}"
    exit 1
fi

# Run inference script
log "Starting DCGAN inference and evaluation..."
python3 "${INFERENCE_SCRIPT}" --checkpoint "${FINAL_MODEL}" --evaluate --interpolate --output_dir "${OUTPUT_DIR}" > "${INFERENCE_LOG}" 2>&1
INFERENCE_EXIT_CODE=$?

if [ ${INFERENCE_EXIT_CODE} -ne 0 ]; then
    log "Error: Inference failed with exit code ${INFERENCE_EXIT_CODE}. See log: ${INFERENCE_LOG}"
    exit ${INFERENCE_EXIT_CODE}
fi

log "Inference and evaluation completed successfully"
log "Results saved to ${OUTPUT_DIR}"

# Create a summary of results
SUMMARY_FILE="${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
echo "DCGAN Training and Inference Summary (${TIMESTAMP})" > "${SUMMARY_FILE}"
echo "=================================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "Training completed at: $(date)" >> "${SUMMARY_FILE}"
echo "Training log: ${TRAIN_LOG}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "Inference completed at: $(date)" >> "${SUMMARY_FILE}"
echo "Inference log: ${INFERENCE_LOG}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Extract FID score if available
if [ -f "${OUTPUT_DIR}/evaluation_results.txt" ]; then
    echo "Evaluation Results:" >> "${SUMMARY_FILE}"
    cat "${OUTPUT_DIR}/evaluation_results.txt" >> "${SUMMARY_FILE}"
else
    echo "No evaluation results found." >> "${SUMMARY_FILE}"
fi

log "Summary created at ${SUMMARY_FILE}"

# Upload results to S3
log "Uploading results to S3 bucket: ${S3_BUCKET}"

# Create a metadata file with instance information
METADATA_FILE="/tmp/dcgan_metadata_${TIMESTAMP}.json"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null | sed 's/[a-z]$//' || echo "unknown")

cat > "${METADATA_FILE}" << EOL
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "instance_id": "${INSTANCE_ID}",
    "instance_type": "${INSTANCE_TYPE}",
    "region": "${REGION}",
    "run_id": "${TIMESTAMP}"
}
EOL

# Set S3 path with date prefix
S3_DATE_PREFIX="${S3_PREFIX}/$(date +%Y%m%d)/${TIMESTAMP}"

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    log "Warning: AWS CLI not found. Skipping S3 upload."
else
    # Upload metadata
    log "Uploading instance metadata" | tee -a "${UPLOAD_LOG}"
    aws s3 cp "${METADATA_FILE}" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/metadata.json" >> "${UPLOAD_LOG}" 2>&1

    # Upload logs
    log "Uploading logs" | tee -a "${UPLOAD_LOG}"
    aws s3 cp "${TRAIN_LOG}" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/logs/train.log" >> "${UPLOAD_LOG}" 2>&1
    aws s3 cp "${INFERENCE_LOG}" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/logs/inference.log" >> "${UPLOAD_LOG}" 2>&1
    aws s3 cp "${SUMMARY_FILE}" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/logs/summary.txt" >> "${UPLOAD_LOG}" 2>&1

    # Upload final model
    log "Uploading trained model" | tee -a "${UPLOAD_LOG}"
    aws s3 cp "${FINAL_MODEL}" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/models/final_model.pth" >> "${UPLOAD_LOG}" 2>&1

    # Upload sample images
    log "Uploading generated samples" | tee -a "${UPLOAD_LOG}"
    aws s3 cp "${OUTPUT_DIR}/generated_samples.png" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/images/generated_samples.png" >> "${UPLOAD_LOG}" 2>&1
    aws s3 cp "${OUTPUT_DIR}/latent_interpolation.png" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/images/latent_interpolation.png" >> "${UPLOAD_LOG}" 2>&1

    # Upload final epoch samples
    FINAL_EPOCH_SAMPLE=$(ls -t ${SAMPLES_DIR}/samples_epoch_*.png | head -1)
    if [ -f "${FINAL_EPOCH_SAMPLE}" ]; then
        aws s3 cp "${FINAL_EPOCH_SAMPLE}" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/images/final_epoch_samples.png" >> "${UPLOAD_LOG}" 2>&1
    fi

    # Upload evaluation results
    if [ -f "${OUTPUT_DIR}/evaluation_results.txt" ]; then
        aws s3 cp "${OUTPUT_DIR}/evaluation_results.txt" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/evaluation_results.txt" >> "${UPLOAD_LOG}" 2>&1
    fi

    # Upload this upload log itself
    aws s3 cp "${UPLOAD_LOG}" "s3://${S3_BUCKET}/${S3_DATE_PREFIX}/logs/s3_upload.log" >> "${UPLOAD_LOG}" 2>&1

    log "S3 upload completed. Results available at: s3://${S3_BUCKET}/${S3_DATE_PREFIX}/"
else
    log "AWS CLI not found - skipping S3 upload"
fi

log "Job completed successfully"

exit 0
EOF

# Replace placeholders in the script
sed -i "s|__TRAIN_SCRIPT__|${TRAIN_SCRIPT}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__INFERENCE_SCRIPT__|${INFERENCE_SCRIPT}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__LOG_DIR__|${LOG_DIR}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__CHECKPOINT_DIR__|${CHECKPOINT_DIR}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__OUTPUT_DIR__|${OUTPUT_DIR}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__SAMPLES_DIR__|${SAMPLES_DIR}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__VENV_PATH__|${VENV_PATH}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__S3_BUCKET__|${S3_BUCKET}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__S3_PREFIX__|${S3_PREFIX}|g" "${WRAPPER_SCRIPT}"

# Make the wrapper script executable
chmod +x "${WRAPPER_SCRIPT}"

# Create a cron job entry file
CRON_FILE="/tmp/dcgan_cron"

# Add header to cron file
echo "# DCGAN training and inference scheduled job" > "${CRON_FILE}"

# Ask user for scheduling preferences
echo "Setting up cron job for DCGAN training and inference..."
echo "Please choose a scheduling option:"
echo "1. Run daily (specify time)"
echo "2. Run weekly (specify day and time)"
echo "3. Run monthly (specify day and time)"
echo "4. Custom schedule (specify cron expression)"
read -p "Enter your choice (1-4): " SCHEDULE_CHOICE

case ${SCHEDULE_CHOICE} in
    1)
        read -p "Enter time to run daily (HH:MM in 24-hour format): " RUN_TIME
        HOUR=$(echo ${RUN_TIME} | cut -d: -f1)
        MINUTE=$(echo ${RUN_TIME} | cut -d: -f2)
        echo "${MINUTE} ${HOUR} * * * ${WRAPPER_SCRIPT}" >> "${CRON_FILE}"
        SCHEDULE_DESC="daily at ${RUN_TIME}"
        ;;
    2)
        read -p "Enter day of week (0-6, where 0 is Sunday): " DOW
        read -p "Enter time to run (HH:MM in 24-hour format): " RUN_TIME
        HOUR=$(echo ${RUN_TIME} | cut -d: -f1)
        MINUTE=$(echo ${RUN_TIME} | cut -d: -f2)
        echo "${MINUTE} ${HOUR} * * ${DOW} ${WRAPPER_SCRIPT}" >> "${CRON_FILE}"
        SCHEDULE_DESC="weekly on day ${DOW} at ${RUN_TIME}"
        ;;
    3)
        read -p "Enter day of month (1-31): " DOM
        read -p "Enter time to run (HH:MM in 24-hour format): " RUN_TIME
        HOUR=$(echo ${RUN_TIME} | cut -d: -f1)
        MINUTE=$(echo ${RUN_TIME} | cut -d: -f2)
        echo "${MINUTE} ${HOUR} ${DOM} * * ${WRAPPER_SCRIPT}" >> "${CRON_FILE}"
        SCHEDULE_DESC="monthly on day ${DOM} at ${RUN_TIME}"
        ;;
    4)
        read -p "Enter custom cron expression (e.g. '0 2 * * 0'): " CRON_EXPR
        echo "${CRON_EXPR} ${WRAPPER_SCRIPT}" >> "${CRON_FILE}"
        SCHEDULE_DESC="custom schedule: ${CRON_EXPR}"
        ;;
    *)
        echo "Invalid choice. Using default: daily at midnight."
        echo "0 0 * * * ${WRAPPER_SCRIPT}" >> "${CRON_FILE}"
        SCHEDULE_DESC="daily at midnight (default)"
        ;;
esac

# Add the cron job to the user's crontab
crontab -l > /tmp/current_cron 2>/dev/null || true
cat "${CRON_FILE}" >> /tmp/current_cron
crontab /tmp/current_cron
rm /tmp/current_cron /tmp/dcgan_cron

# Verify AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Warning: AWS CLI is not installed. S3 uploads will not work."
    echo "To install AWS CLI, run: pip install awscli"
    echo "Then configure with: aws configure"
else
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "Warning: AWS credentials are not configured or invalid."
        echo "Please run 'aws configure' to set up your AWS credentials."
    else
        echo "AWS credentials verified successfully."
        
        # Check if S3 bucket exists
        if ! aws s3 ls "s3://${S3_BUCKET}" &> /dev/null; then
            echo "Warning: S3 bucket '${S3_BUCKET}' does not exist or you don't have access to it."
            read -p "Would you like to create this bucket? (y/n): " CREATE_BUCKET
            if [[ "${CREATE_BUCKET}" == "y" || "${CREATE_BUCKET}" == "Y" ]]; then
                aws s3 mb "s3://${S3_BUCKET}"
                echo "S3 bucket created: ${S3_BUCKET}"
            else
                echo "Please create the S3 bucket manually or use an existing bucket."
            fi
        else
            echo "S3 bucket '${S3_BUCKET}' verified successfully."
        fi
    fi
fi

echo ""
echo "Cron job setup complete!"
echo "Your DCGAN training and inference will run ${SCHEDULE_DESC}"
echo "Results will be uploaded to S3 bucket: ${S3_BUCKET}/${S3_PREFIX}/[date]/[timestamp]/"
echo "Wrapper script: ${WRAPPER_SCRIPT}"
echo "Logs will be stored in: ${LOG_DIR}"
echo "To edit the schedule later, use 'crontab -e'"

# Optional: Run the job manually now
read -p "Do you want to run the job now for testing? (y/n): " RUN_NOW
if [[ "${RUN_NOW}" == "y" || "${RUN_NOW}" == "Y" ]]; then
    echo "Running the job now..."
    ${WRAPPER_SCRIPT}
else
    echo "Job scheduled but not running now. It will run according to the schedule."
fi
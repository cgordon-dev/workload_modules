#!/bin/bash
# DCGAN Cron Job Setup Script
# This script sets up a cron job to run DCGAN training and inference on a schedule

# Directory paths (update these to match your environment)
PROJECT_DIR="/home/ubuntu/gpu-monitoring/gpu_monitoring_project/dcgan_model_v2"
TRAIN_SCRIPT="${PROJECT_DIR}/dcgan_train.py"
INFERENCE_SCRIPT="${PROJECT_DIR}/fixed_dcgan_inference.py"
LOG_DIR="${PROJECT_DIR}/logs"
CHECKPOINT_DIR="${PROJECT_DIR}/models"
OUTPUT_DIR="${PROJECT_DIR}/output"
VENV_PATH="/home/ubuntu/gpu-monitoring/venv/bin/activate"

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
VENV_PATH="__VENV_PATH__"

# Create timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRAIN_LOG="${LOG_DIR}/train_${TIMESTAMP}.log"
INFERENCE_LOG="${LOG_DIR}/inference_${TIMESTAMP}.log"
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

# Optional: Send notification that the job is complete
# For example, if you have mail configured:
# echo "DCGAN training and inference completed successfully. Results in ${OUTPUT_DIR}" | mail -s "DCGAN Job Complete" your@email.com

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
log "Job completed successfully"

exit 0
EOF

# Replace placeholders in the script
sed -i "s|__TRAIN_SCRIPT__|${TRAIN_SCRIPT}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__INFERENCE_SCRIPT__|${INFERENCE_SCRIPT}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__LOG_DIR__|${LOG_DIR}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__CHECKPOINT_DIR__|${CHECKPOINT_DIR}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__OUTPUT_DIR__|${OUTPUT_DIR}|g" "${WRAPPER_SCRIPT}"
sed -i "s|__VENV_PATH__|${VENV_PATH}|g" "${WRAPPER_SCRIPT}"

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

echo "Cron job setup complete!"
echo "Your DCGAN training and inference will run ${SCHEDULE_DESC}"
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

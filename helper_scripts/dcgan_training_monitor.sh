#!/bin/bash
# GPU Training Monitor Script for EC2 g4dn Instances
# This script launches your DCGAN training with monitoring and auto-recovery

set -e  # Exit on any error

# Configuration
PROJECT_DIR="$HOME/gpu-monitoring/gpu_monitoring_project/dcgan_model_v2"
LOG_FILE="$PROJECT_DIR/training_log_$(date +%Y%m%d_%H%M%S).log"
CHECKPOINT_DIR="$PROJECT_DIR/models"
CHECKPOINT_FILE="best_model.pth"
MAX_RETRIES=3
MONITOR_INTERVAL=30  # seconds
THRESHOLD_GPU_UTIL=90  # percentage
THRESHOLD_GPU_MEM=90   # percentage
EMAIL=""  # Set your email to receive notifications

# Check if we're in a tmux session, if not start one
if [ -z "$TMUX" ]; then
    echo "Starting a new tmux session for GPU training..."
    tmux new-session -d -s dcgan_training "$(readlink -f $0)"
    echo "Training session started in tmux. Attach with: tmux attach -t dcgan_training"
    echo "View logs with: tail -f $LOG_FILE"
    exit 0
fi

# Create log directory if it doesn't exist
mkdir -p $(dirname "$LOG_FILE")

# Log function
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

# Send email notification
send_notification() {
    local subject="$1"
    local message="$2"
    
    if [ -n "$EMAIL" ]; then
        echo "$message" | mail -s "$subject" "$EMAIL"
        log "Notification sent to $EMAIL"
    fi
}

# Check if nvidia-smi is available
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log "ERROR: nvidia-smi not found. Is the NVIDIA driver installed?"
        exit 1
    fi
    
    # Check if GPU is visible
    if ! nvidia-smi &> /dev/null; then
        log "ERROR: Unable to communicate with GPU."
        exit 1
    fi
    
    # Get GPU info
    log "GPU Information:"
    nvidia-smi | tee -a "$LOG_FILE"
}

# Monitor GPU usage
monitor_gpu() {
    local pid=$1
    local start_time=$(date +%s)
    local duration=0
    local gpu_stats=""
    
    log "Starting GPU monitoring for process $pid"
    
    # Create stats file
    local stats_file="$PROJECT_DIR/gpu_stats_$(date +%Y%m%d_%H%M%S).csv"
    echo "timestamp,gpu_utilization,gpu_memory_used,gpu_memory_total,gpu_temperature,gpu_power" > "$stats_file"
    
    while kill -0 $pid 2>/dev/null; do
        # Get current time
        local current_time=$(date +%s)
        duration=$((current_time - start_time))
        
        # Get GPU utilization
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        gpu_mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        gpu_mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        gpu_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
        
        # Calculate GPU memory percentage
        gpu_mem_percent=$(echo "scale=2; $gpu_mem_used * 100 / $gpu_mem_total" | bc)
        
        # Save stats
        echo "$(date +%Y-%m-%d_%H:%M:%S),$gpu_util,$gpu_mem_used,$gpu_mem_total,$gpu_temp,$gpu_power" >> "$stats_file"
        
        # Check if GPU utilization or memory exceeds thresholds
        if (( $(echo "$gpu_util > $THRESHOLD_GPU_UTIL" | bc -l) )); then
            log "WARNING: GPU utilization at $gpu_util% (threshold: $THRESHOLD_GPU_UTIL%)"
        fi
        
        if (( $(echo "$gpu_mem_percent > $THRESHOLD_GPU_MEM" | bc -l) )); then
            log "WARNING: GPU memory at $gpu_mem_percent% (threshold: $THRESHOLD_GPU_MEM%)"
        fi
        
        # Show progress every 5 minutes
        if (( duration % 300 == 0 )); then
            hours=$((duration / 3600))
            minutes=$(( (duration % 3600) / 60 ))
            seconds=$((duration % 60))
            
            log "Training running for ${hours}h ${minutes}m ${seconds}s | GPU: ${gpu_util}% | Memory: ${gpu_mem_used}MB/${gpu_mem_total}MB | Temp: ${gpu_temp}°C | Power: ${gpu_power}W"
        fi
        
        sleep $MONITOR_INTERVAL
    done
    
    # Calculate training time
    hours=$((duration / 3600))
    minutes=$(( (duration % 3600) / 60 ))
    seconds=$((duration % 60))
    
    log "Training completed. Total time: ${hours}h ${minutes}m ${seconds}s"
    
    # Plot GPU stats if gnuplot is available
    if command -v gnuplot &> /dev/null; then
        log "Generating GPU usage graphs..."
        
        # Create gnuplot script
        cat << EOF > "$PROJECT_DIR/plot_gpu.gnuplot"
set terminal png size 1200,800
set output '$PROJECT_DIR/gpu_utilization.png'
set title 'GPU Utilization Over Time'
set xlabel 'Time'
set ylabel 'Utilization (%)'
set grid
set xdata time
set timefmt '%Y-%m-%d_%H:%M:%S'
set format x '%H:%M:%S'
plot '$stats_file' using 1:2 with lines title 'GPU Utilization (%)'

set output '$PROJECT_DIR/gpu_memory.png'
set title 'GPU Memory Usage Over Time'
set ylabel 'Memory (MB)'
plot '$stats_file' using 1:3 with lines title 'GPU Memory Used (MB)'

set output '$PROJECT_DIR/gpu_temperature.png'
set title 'GPU Temperature Over Time'
set ylabel 'Temperature (°C)'
plot '$stats_file' using 1:5 with lines title 'GPU Temperature (°C)'

set output '$PROJECT_DIR/gpu_power.png'
set title 'GPU Power Usage Over Time'
set ylabel 'Power (W)'
plot '$stats_file' using 1:6 with lines title 'GPU Power (W)'
EOF
        
        # Run gnuplot
        gnuplot "$PROJECT_DIR/plot_gpu.gnuplot"
        log "GPU usage graphs saved to $PROJECT_DIR"
    fi
}

# Run the training with monitoring
run_training() {
    local retry_count=0
    local training_succeeded=false
    
    while [ $retry_count -lt $MAX_RETRIES ] && [ "$training_succeeded" = "false" ]; do
        log "Starting training job (attempt $((retry_count + 1)) of $MAX_RETRIES)"
        
        # Check for existing checkpoint
        if [ -f "$CHECKPOINT_DIR/$CHECKPOINT_FILE" ] && [ $retry_count -gt 0 ]; then
            log "Found existing checkpoint, will resume training"
            export RESUME_FROM="$CHECKPOINT_DIR/$CHECKPOINT_FILE"
        fi
        
        # Navigate to project directory
        cd "$PROJECT_DIR"
        
        # Start the training in background and capture its PID
        (./run_dcgan.sh >> "$LOG_FILE" 2>&1) &
        TRAIN_PID=$!
        
        # Start GPU monitoring
        monitor_gpu $TRAIN_PID &
        MONITOR_PID=$!
        
        # Wait for training to complete
        wait $TRAIN_PID
        TRAIN_EXIT_CODE=$?
        
        # Stop monitoring
        kill $MONITOR_PID 2>/dev/null || true
        wait $MONITOR_PID 2>/dev/null || true
        
        # Check if training was successful
        if [ $TRAIN_EXIT_CODE -eq 0 ]; then
            log "Training completed successfully!"
            training_succeeded=true
            
            if [ -f "$CHECKPOINT_DIR/$CHECKPOINT_FILE" ]; then
                log "Model saved to $CHECKPOINT_DIR/$CHECKPOINT_FILE"
                send_notification "DCGAN Training Completed" "Training completed successfully.\nModel saved to $CHECKPOINT_DIR/$CHECKPOINT_FILE"
            else
                log "WARNING: Training completed but no model file found."
                send_notification "DCGAN Training Completed (Warning)" "Training completed but no model file was found at $CHECKPOINT_DIR/$CHECKPOINT_FILE"
            fi
        else
            retry_count=$((retry_count + 1))
            log "Training failed with exit code $TRAIN_EXIT_CODE"
            
            if [ $retry_count -lt $MAX_RETRIES ]; then
                log "Retrying in 60 seconds..."
                sleep 60
            else
                log "Maximum retry attempts reached. Training failed."
                send_notification "DCGAN Training Failed" "Training failed after $MAX_RETRIES attempts.\nCheck logs at $LOG_FILE"
            fi
        fi
    done
    
    return $TRAIN_EXIT_CODE
}

# Main function
main() {
    log "======= DCGAN Training Monitor ======="
    log "Project directory: $PROJECT_DIR"
    log "Log file: $LOG_FILE"
    
    # Check for NVIDIA GPU
    check_gpu
    
    # Install required packages if missing
    if ! command -v bc &> /dev/null; then
        log "Installing bc for calculations..."
        sudo apt-get update && sudo apt-get install -y bc
    fi
    
    if ! command -v gnuplot &> /dev/null; then
        log "Installing gnuplot for performance graphs..."
        sudo apt-get update && sudo apt-get install -y gnuplot
    fi
    
    # Check if tmux is installed
    if ! command -v tmux &> /dev/null; then
        log "Installing tmux for persistent sessions..."
        sudo apt-get update && sudo apt-get install -y tmux
    fi
    
    # Run training
    run_training
    RESULT=$?
    
    # Run inference if training succeeded
    if [ $RESULT -eq 0 ]; then
        log "Running inference and evaluation..."
        cd "$PROJECT_DIR"
        
        # Run inference script
        python3 dcgan_inference.py --checkpoint "$CHECKPOINT_DIR/$CHECKPOINT_FILE" \
                                   --evaluate --interpolate \
                                   --output_dir "./output" >> "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            log "Inference and evaluation completed successfully!"
            send_notification "DCGAN Inference Completed" "Inference and evaluation completed successfully.\nResults saved to $PROJECT_DIR/output"
        else
            log "Inference failed."
            send_notification "DCGAN Inference Failed" "Inference failed.\nCheck logs at $LOG_FILE"
        fi
    fi
    
    log "======= Training Monitor Finished ======="
    return $RESULT
}

# Start main function
main

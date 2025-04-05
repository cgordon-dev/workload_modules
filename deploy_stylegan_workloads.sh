#!/bin/bash
#
# StyleGAN2-ADA AWS EC2 Spot Instance Deployment Script
# 
# This script automates deployment and scheduling of StyleGAN2-ADA workloads
# across multiple AWS EC2 Spot instance types with different NVIDIA GPUs.
#
# It handles:
# - Launching EC2 spot instances (g4dn.xlarge, p3.2xlarge, p4d.24xlarge)
# - Setting up StyleGAN2-ADA environment on each instance
# - Running training, inference and optimization workloads in sequence
# - Collecting GPU telemetry with Prometheus and Grafana
# - Storing logs in S3 by instance type
#

set -e  # Exit on error

# =============================================================================
# Configuration Variables - Edit these as needed
# =============================================================================

# AWS Configuration
AWS_PROFILE="default"                              # AWS CLI profile
AWS_REGION="us-east-1"                             # AWS region
KEY_PAIR_NAME="your-key-pair-name"                 # EC2 key pair name
SECURITY_GROUP_ID="sg-0123456789abcdef0"           # Security Group ID
SUBNET_ID="subnet-0123456789abcdef0"               # Subnet ID
IAM_INSTANCE_PROFILE="EC2InstanceProfileWithS3"    # IAM role for EC2 with S3 access
AMI_ID="ami-0123456789abcdef0"                     # Ubuntu 20.04 with CUDA 11.4 AMI

# GitHub Repository
GITHUB_REPO="https://github.com/yourusername/stylegan2-ada-pytorch.git"
REPO_BRANCH="main"

# S3 Bucket for logs
S3_BUCKET="your-stylegan-logs-bucket"

# Dataset Configuration
DATASET_NAME="fashion-mnist"
DATASET_S3_PATH="s3://your-datasets-bucket/fashion-mnist"

# Schedule Configuration
EXECUTION_INTERVAL_HOURS=24                        # Run sequence every X hours

# Instance Types with their specific configurations
declare -A INSTANCE_CONFIGS=(
  ["g4dn.xlarge"]="--image-size=256 --batch=16 --mixed-precision-mode=aggressive"
  ["p3.2xlarge"]="--image-size=512 --batch=32 --mixed-precision-mode=default"
  ["p4d.24xlarge"]="--image-size=1024 --batch=64 --mixed-precision-mode=conservative"
)

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Launch EC2 Spot Instance and return instance ID
launch_spot_instance() {
    local instance_type=$1
    local availability_zone=$2
    
    log "Requesting $instance_type spot instance in $availability_zone..."
    
    # Create spot instance request
    instance_id=$(aws ec2 request-spot-instances \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --spot-price "auto" \
        --instance-count 1 \
        --type "one-time" \
        --launch-specification "{
            \"ImageId\": \"$AMI_ID\",
            \"InstanceType\": \"$instance_type\",
            \"KeyName\": \"$KEY_PAIR_NAME\",
            \"SecurityGroupIds\": [\"$SECURITY_GROUP_ID\"],
            \"SubnetId\": \"$SUBNET_ID\",
            \"IamInstanceProfile\": {\"Name\": \"$IAM_INSTANCE_PROFILE\"},
            \"BlockDeviceMappings\": [
                {
                    \"DeviceName\": \"/dev/sda1\",
                    \"Ebs\": {
                        \"VolumeSize\": 100,
                        \"VolumeType\": \"gp3\",
                        \"DeleteOnTermination\": true
                    }
                }
            ]
        }" \
        --query "SpotInstanceRequests[0].InstanceId" \
        --output text)
    
    # Wait for instance to be running
    log "Waiting for instance $instance_id to be running..."
    aws ec2 wait instance-running \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --instance-ids "$instance_id"
    
    # Get instance public IP
    public_ip=$(aws ec2 describe-instances \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --instance-ids "$instance_id" \
        --query "Reservations[0].Instances[0].PublicIpAddress" \
        --output text)
    
    log "Instance $instance_id ($instance_type) is running at $public_ip"
    echo "$instance_id:$public_ip"
}

# Setup instance with StyleGAN2-ADA dependencies
setup_instance() {
    local instance_ip=$1
    local instance_type=$2
    local ssh_key="$HOME/.ssh/$KEY_PAIR_NAME.pem"
    
    log "Setting up instance $instance_ip ($instance_type)..."
    
    # Wait for SSH to be available
    while ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$ssh_key" ubuntu@"$instance_ip" echo "SSH available"; do
        log "Waiting for SSH on $instance_ip..."
        sleep 10
    done
    
    # Install dependencies and clone repository
    ssh -o StrictHostKeyChecking=no -i "$ssh_key" ubuntu@"$instance_ip" << EOF
        set -e
        
        # Update and install dependencies
        echo "Installing system dependencies..."
        sudo apt-get update
        sudo apt-get install -y git python3-pip python3-dev build-essential libssl-dev libffi-dev libxml2-dev \
            libxslt1-dev zlib1g-dev python3-setuptools unzip ninja-build libjpeg-dev libpng-dev wget

        # Setup Conda
        echo "Setting up Conda environment..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        conda init bash
        source ~/.bashrc
        
        # Create and activate conda environment for StyleGAN2
        conda create -y -n stylegan python=3.8
        conda activate stylegan
        
        # Clone StyleGAN2-ADA repository
        echo "Cloning repository..."
        git clone $GITHUB_REPO --branch $REPO_BRANCH
        cd stylegan2-ada-pytorch
        
        # Install requirements
        pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
        pip install ninja imageio-ffmpeg==0.4.3 pyspng lpips click requests tqdm pyrallis pyrsistent pydantic
        pip install ftfy regex tqdm pillow matplotlib boto3 prometheus-client
        
        # Install StyleGAN2 requirements
        pip install -r requirements.txt
        
        # Download dataset
        echo "Downloading dataset $DATASET_NAME..."
        mkdir -p datasets
        aws s3 cp $DATASET_S3_PATH datasets/$DATASET_NAME --recursive
        
        # Setup Prometheus for telemetry
        echo "Setting up Prometheus and Node Exporter..."
        wget https://github.com/prometheus/prometheus/releases/download/v2.38.0/prometheus-2.38.0.linux-amd64.tar.gz
        tar xvfz prometheus-2.38.0.linux-amd64.tar.gz
        
        wget https://github.com/prometheus/node_exporter/releases/download/v1.4.0/node_exporter-1.4.0.linux-amd64.tar.gz
        tar xvfz node_exporter-1.4.0.linux-amd64.tar.gz
        
        # Install NVIDIA DCGM Exporter for GPU metrics
        distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID | sed -e 's/\.//g')
        wget https://developer.download.nvidia.com/compute/cuda/repos/\$distribution/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        sudo apt-get install -y datacenter-gpu-manager
        sudo apt-get install -y dcgm-exporter
        
        # Create a prometheus.yml configuration
        cat > prometheus-2.38.0.linux-amd64/prometheus.yml << 'PROMEOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
  - job_name: 'dcgm'
    static_configs:
      - targets: ['localhost:9400']
PROMEOF
        
        # Create directories for logs
        mkdir -p logs/training logs/inference logs/optimization
        
        echo "Setup complete on $instance_type instance!"
EOF
    
    log "Instance $instance_ip ($instance_type) setup completed!"
}

# Start telemetry services
start_telemetry() {
    local instance_ip=$1
    local ssh_key="$HOME/.ssh/$KEY_PAIR_NAME.pem"
    
    log "Starting telemetry services on $instance_ip..."
    
    ssh -o StrictHostKeyChecking=no -i "$ssh_key" ubuntu@"$instance_ip" << EOF
        set -e
        
        # Start Node Exporter
        nohup ./node_exporter-1.4.0.linux-amd64/node_exporter > node_exporter.log 2>&1 &
        
        # Start DCGM Exporter for GPU metrics
        sudo systemctl start dcgm-exporter
        
        # Start Prometheus
        cd prometheus-2.38.0.linux-amd64
        nohup ./prometheus --config.file=prometheus.yml > prometheus.log 2>&1 &
        
        echo "Telemetry services started!"
EOF
}

# Run workload sequence
run_workload_sequence() {
    local instance_ip=$1
    local instance_type=$2
    local config_args=${INSTANCE_CONFIGS[$instance_type]}
    local ssh_key="$HOME/.ssh/$KEY_PAIR_NAME.pem"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_dir="logs/$instance_type/$timestamp"
    
    log "Starting workload sequence on $instance_ip ($instance_type)..."
    
    ssh -o StrictHostKeyChecking=no -i "$ssh_key" ubuntu@"$instance_ip" << EOF
        set -e
        
        # Activate conda environment
        source ~/.bashrc
        conda activate stylegan
        
        cd stylegan2-ada-pytorch/workload_modules
        
        # Create results directory with timestamp
        results_dir="../results/$instance_type/run_$timestamp"
        mkdir -p "\$results_dir"
        
        # 1. Training workload
        echo "Starting training workload..."
        python training_workload.py \
            --dataset-path=../datasets/$DATASET_NAME \
            --outdir="\$results_dir/training" \
            --industry=fashion \
            $config_args \
            --kimg=1000 \
            --mirror=true \
            > "../logs/training/${instance_type}_${timestamp}.log" 2>&1
        
        # Get the latest network pkl
        latest_network=\$(ls -t "\$results_dir/training/"*network*.pkl | head -1)
        echo "Training complete. Latest network: \$latest_network"
        
        # 2. Inference workload
        echo "Starting inference workload..."
        python inference_workload.py \
            --network="\$latest_network" \
            --outdir="\$results_dir/inference" \
            --seeds=0-10 \
            --industry=fashion \
            --trunc=0.7 \
            > "../logs/inference/${instance_type}_${timestamp}.log" 2>&1
        
        # 3a. Fine-tuning optimization
        echo "Starting fine-tuning optimization..."
        python fine_tuning_optimization.py \
            --dataset-path=../datasets/$DATASET_NAME \
            --resume="\$latest_network" \
            --outdir="\$results_dir/fine_tuning" \
            --freezed=6 \
            --kimg=500 \
            > "../logs/optimization/${instance_type}_ft_${timestamp}.log" 2>&1
        
        # 3b. Mixed precision optimization
        echo "Starting mixed precision optimization..."
        python mixed_precision_optimization.py \
            --dataset-path=../datasets/$DATASET_NAME \
            --outdir="\$results_dir/mixed_precision" \
            $config_args \
            --kimg=500 \
            > "../logs/optimization/${instance_type}_mp_${timestamp}.log" 2>&1
        
        # 3c. Latent vector optimization
        # First, use some generated images as targets
        mkdir -p "\$results_dir/targets"
        cp "\$results_dir/inference/seed0000.png" "\$results_dir/targets/"
        cp "\$results_dir/inference/seed0001.png" "\$results_dir/targets/"
        
        echo "Starting latent vector optimization..."
        python latent_vector_optimization.py \
            --network="\$latest_network" \
            --target="\$results_dir/targets" \
            --outdir="\$results_dir/latent_optimization" \
            --num-steps=500 \
            --latent-space="w+" \
            > "../logs/optimization/${instance_type}_lv_${timestamp}.log" 2>&1
        
        # Collect and upload all logs and results to S3
        echo "Uploading logs and results to S3..."
        aws s3 cp ../logs s3://$S3_BUCKET/logs/$instance_type/ --recursive
        aws s3 cp "\$results_dir" s3://$S3_BUCKET/results/$instance_type/run_$timestamp/ --recursive
        
        echo "Workload sequence completed on $instance_type!"
EOF
    
    log "Workload sequence completed on $instance_ip ($instance_type)!"
}

# Collect metrics from Prometheus and upload to S3
collect_telemetry() {
    local instance_ip=$1
    local instance_type=$2
    local ssh_key="$HOME/.ssh/$KEY_PAIR_NAME.pem"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    log "Collecting telemetry from $instance_ip ($instance_type)..."
    
    ssh -o StrictHostKeyChecking=no -i "$ssh_key" ubuntu@"$instance_ip" << EOF
        set -e
        
        # Export metrics from Prometheus
        mkdir -p telemetry
        
        # Get GPU utilization metrics
        curl -s "http://localhost:9090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL" > telemetry/gpu_utilization.json
        
        # Get GPU memory usage
        curl -s "http://localhost:9090/api/v1/query?query=DCGM_FI_DEV_FB_USED" > telemetry/gpu_memory_used.json
        
        # Get GPU temperature
        curl -s "http://localhost:9090/api/v1/query?query=DCGM_FI_DEV_GPU_TEMP" > telemetry/gpu_temperature.json
        
        # Get CPU usage
        curl -s "http://localhost:9090/api/v1/query?query=node_cpu_seconds_total" > telemetry/cpu_usage.json
        
        # Get memory usage
        curl -s "http://localhost:9090/api/v1/query?query=node_memory_MemTotal_bytes-node_memory_MemFree_bytes-node_memory_Cached_bytes" > telemetry/memory_usage.json
        
        # Upload telemetry to S3
        aws s3 cp telemetry s3://$S3_BUCKET/telemetry/$instance_type/$timestamp/ --recursive
        
        echo "Telemetry collected and uploaded to S3!"
EOF
    
    log "Telemetry collected from $instance_ip ($instance_type)!"
}

# Terminate EC2 instance
terminate_instance() {
    local instance_id=$1
    
    log "Terminating instance $instance_id..."
    
    aws ec2 terminate-instances \
        --profile "$AWS_PROFILE" \
        --region "$AWS_REGION" \
        --instance-ids "$instance_id"
    
    log "Instance $instance_id terminated!"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log "Starting StyleGAN2-ADA workload automation on AWS EC2 Spot instances..."
    
    # Instance types to launch
    instance_types=("g4dn.xlarge" "p3.2xlarge" "p4d.24xlarge")
    
    # Main execution loop
    while true; do
        log "Starting new execution cycle..."
        
        # Track instance IDs and IPs for this cycle
        declare -A instances
        
        # Launch instances
        for instance_type in "${instance_types[@]}"; do
            instance_data=$(launch_spot_instance "$instance_type" "${AWS_REGION}a")
            instance_id=$(echo "$instance_data" | cut -d':' -f1)
            instance_ip=$(echo "$instance_data" | cut -d':' -f2)
            instances["$instance_type"]="$instance_id:$instance_ip"
            
            # Setup instance
            setup_instance "$instance_ip" "$instance_type"
            
            # Start telemetry services
            start_telemetry "$instance_ip"
        done
        
        # Run workloads in parallel on all instances
        for instance_type in "${instance_types[@]}"; do
            instance_data="${instances[$instance_type]}"
            instance_id=$(echo "$instance_data" | cut -d':' -f1)
            instance_ip=$(echo "$instance_data" | cut -d':' -f2)
            
            # Run workload sequence in background
            (run_workload_sequence "$instance_ip" "$instance_type") &
        done
        
        # Wait for all background jobs to complete
        wait
        
        # Collect telemetry from all instances
        for instance_type in "${instance_types[@]}"; do
            instance_data="${instances[$instance_type]}"
            instance_id=$(echo "$instance_data" | cut -d':' -f1)
            instance_ip=$(echo "$instance_data" | cut -d':' -f2)
            
            collect_telemetry "$instance_ip" "$instance_type"
        done
        
        # Terminate instances
        for instance_type in "${instance_types[@]}"; do
            instance_data="${instances[$instance_type]}"
            instance_id=$(echo "$instance_data" | cut -d':' -f1)
            
            terminate_instance "$instance_id"
        done
        
        log "Execution cycle completed!"
        
        # Sleep until next cycle
        log "Sleeping for $EXECUTION_INTERVAL_HOURS hours until next cycle..."
        sleep "$((EXECUTION_INTERVAL_HOURS * 3600))"
    done
}

# Run main function
main
# StyleGAN2-ADA AWS EC2 Spot Instance Deployment

This document describes how to use the `deploy_stylegan_workloads.sh` script to automate StyleGAN2-ADA workloads on AWS EC2 Spot instances with different NVIDIA GPUs.

## Prerequisites

Before running the deployment script, ensure you have:

1. **AWS CLI** installed and configured with appropriate credentials
2. **AWS IAM Role** for EC2 with S3 access permissions
3. **SSH Key Pair** for EC2 instances
4. **Security Group** that allows SSH access (port 22)
5. **S3 Buckets** created for storing logs, results, and datasets
6. **Fashion-MNIST dataset** uploaded to your S3 bucket

## Configuration

Edit the following variables in the script to match your environment:

```bash
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
```

## AWS IAM Configuration

Create an IAM role for EC2 instances with the following policies:

1. **AmazonS3FullAccess** - For storing logs and results
2. **AmazonEC2ReadOnlyAccess** - For instance metadata

## AWS Security Group Configuration

Create a security group with the following inbound rules:

1. **SSH (port 22)** - From your IP address
2. **Prometheus (port 9090)** - Optional, for remote monitoring
3. **DCGM Exporter (port 9400)** - Optional, for remote monitoring
4. **Node Exporter (port 9100)** - Optional, for remote monitoring

## Dataset Preparation

1. Download the Fashion-MNIST dataset
2. Upload it to your S3 bucket in the appropriate format (PNG images)
3. Update the `DATASET_S3_PATH` variable in the script

## Instance Types

The script launches three types of EC2 spot instances:

1. **g4dn.xlarge** - NVIDIA T4 GPU
2. **p3.2xlarge** - NVIDIA V100 GPU
3. **p4d.24xlarge** - NVIDIA A100 GPU

Each instance type has specific configurations for the workloads, defined in the `INSTANCE_CONFIGS` variable.

## Workload Sequence

On each instance, the following workloads are executed in sequence:

1. **Training** - Trains a StyleGAN2-ADA model from scratch
2. **Inference** - Generates images using the trained model
3. **Optimization** - Runs various optimization workloads:
   - Fine-tuning optimization
   - Mixed precision optimization
   - Latent vector optimization

## Telemetry Collection

The script collects GPU telemetry data using:

1. **Prometheus** - Time-series database
2. **Node Exporter** - System metrics
3. **DCGM Exporter** - NVIDIA GPU metrics

Metrics collected include:
- GPU utilization
- GPU memory usage
- GPU temperature
- CPU usage
- System memory usage

## Running the Script

1. Make the script executable:
   ```bash
   chmod +x deploy_stylegan_workloads.sh
   ```

2. Run the script:
   ```bash
   ./deploy_stylegan_workloads.sh
   ```

3. The script runs continuously, launching new instances at the specified interval.

4. To stop the script, press `Ctrl+C`.

## Monitoring

All logs and telemetry data are uploaded to your S3 bucket in the following structure:

```
s3://your-stylegan-logs-bucket/
├── logs/
│   ├── g4dn.xlarge/
│   ├── p3.2xlarge/
│   └── p4d.24xlarge/
├── results/
│   ├── g4dn.xlarge/
│   ├── p3.2xlarge/
│   └── p4d.24xlarge/
└── telemetry/
    ├── g4dn.xlarge/
    ├── p3.2xlarge/
    └── p4d.24xlarge/
```

## Customization

To use a different dataset or change workload parameters, edit:

1. The `fashion_mnist_config.yaml` file for general configuration
2. The `INSTANCE_CONFIGS` variable in the script for instance-specific configurations

## Troubleshooting

1. **Instance Launch Failures** - Check AWS EC2 spot instance availability in your region
2. **SSH Connection Issues** - Verify security group rules and key pair
3. **CUDA Errors** - Ensure the AMI has compatible CUDA and driver versions
4. **S3 Access Denied** - Check IAM permissions for the EC2 instance profile
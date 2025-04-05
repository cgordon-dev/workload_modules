#!/bin/bash
#
# Upload Fashion-MNIST dataset to S3 for StyleGAN2-ADA training
#

set -e  # Exit on error

# Configuration - EDIT THESE VARIABLES
AWS_PROFILE="default"                      # AWS CLI profile
S3_BUCKET="your-stylegan-datasets-bucket"  # S3 bucket name
DATASET_NAME="fashion-mnist"               # Dataset name
IMAGE_SIZE=64                              # Image size to resize to

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed. Please install it first."
    exit 1
fi

# Prepare the dataset
echo "Preparing Fashion-MNIST dataset..."
python3 prepare_fashion_mnist.py --output-dir="./datasets/${DATASET_NAME}" --image-size=${IMAGE_SIZE}

# Check if dataset was created
if [ ! -d "./datasets/${DATASET_NAME}" ]; then
    echo "Error: Dataset directory was not created."
    exit 1
fi

# Upload to S3
echo "Uploading dataset to S3..."
aws s3 cp "./datasets/${DATASET_NAME}" "s3://${S3_BUCKET}/${DATASET_NAME}" --recursive --profile ${AWS_PROFILE}

echo "Dataset upload complete!"
echo "S3 path: s3://${S3_BUCKET}/${DATASET_NAME}"
echo ""
echo "Update the DATASET_S3_PATH variable in your deploy_stylegan_workloads.sh script to:"
echo "DATASET_S3_PATH=\"s3://${S3_BUCKET}/${DATASET_NAME}\""
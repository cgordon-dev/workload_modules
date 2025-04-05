#!/bin/bash
# Installation script for continuous StyleGAN2 service
# This script sets up the continuous operation as a system service

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="${SCRIPT_DIR}"

# Set default user and group
DEFAULT_USER="ubuntu"
DEFAULT_GROUP="ubuntu"

# Ask for AWS credentials
echo "Setting up AWS credentials for S3 uploads"
read -p "Enter AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -sp "Enter AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo ""
read -p "Enter AWS Default Region [us-east-1]: " AWS_DEFAULT_REGION
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}

# Ask for user to run the service as
read -p "Enter the user to run the service as [$DEFAULT_USER]: " SERVICE_USER
SERVICE_USER=${SERVICE_USER:-$DEFAULT_USER}

read -p "Enter the group for the service [$DEFAULT_GROUP]: " SERVICE_GROUP
SERVICE_GROUP=${SERVICE_GROUP:-$DEFAULT_GROUP}

# Create the service file
echo "Creating SystemD service file..."
SERVICE_FILE="/etc/systemd/system/stylegan-continuous.service"

cat > ${SERVICE_FILE} << EOL
[Unit]
Description=Continuous StyleGAN2 Training with Automatic S3 Uploads
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PROJECT_DIR}/continuous-stylegan.sh
Restart=always
RestartSec=10
StandardOutput=append:${PROJECT_DIR}/logs/service.log
StandardError=append:${PROJECT_DIR}/logs/service-error.log
# Set environment variables for AWS credentials
Environment="AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}"
Environment="AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}"
Environment="AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}"

[Install]
WantedBy=multi-user.target
EOL

echo "Created service file at ${SERVICE_FILE}"

# Ensure continuous-stylegan.sh script exists and is executable
if [ ! -f "${PROJECT_DIR}/continuous-stylegan.sh" ]; then
    echo "Error: continuous-stylegan.sh script not found at ${PROJECT_DIR}"
    exit 1
fi

chmod +x "${PROJECT_DIR}/continuous-stylegan.sh"

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"
chown -R ${SERVICE_USER}:${SERVICE_GROUP} "${PROJECT_DIR}/logs"

# Reload systemd, enable and start the service
echo "Reloading SystemD daemon..."
systemctl daemon-reload

echo "Enabling StyleGAN continuous service..."
systemctl enable stylegan-continuous.service

echo "Starting StyleGAN continuous service..."
systemctl start stylegan-continuous.service

# Check if service started successfully
sleep 3
if systemctl is-active --quiet stylegan-continuous.service; then
    echo "Service started successfully!"
    echo "You can check the service status with: systemctl status stylegan-continuous.service"
    echo "You can view logs with: journalctl -u stylegan-continuous.service"
else
    echo "Warning: Service may not have started correctly."
    echo "Check status with: systemctl status stylegan-continuous.service"
fi

echo ""
echo "Installation complete!"
echo "The StyleGAN2 continuous operation service will now run in the background"
echo "and automatically upload results to S3 after each run completes."

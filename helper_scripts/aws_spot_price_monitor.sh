#!/bin/bash
# aws_spot_price_monitor.sh
#
# This script retrieves pricing information for AWS g4dn.xlarge and g4dn.2xlarge spot instances,
# logs the data to a file, and uploads the file to an S3 bucket.
# It's designed to be run as a cron job every 30 minutes.
#
# Requirements:
# - AWS CLI installed and configured with appropriate permissions
# - S3 bucket named "aws-gpu-monitoring-logs" must exist
# - Permissions to write to the S3 bucket
#
# Recommended crontab entry:
# */30 * * * * /path/to/aws_spot_price_monitor.sh

# Configuration
LOG_DIR="/tmp/aws-spot-monitoring"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/spot_pricing_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/spot_pricing_latest.log"
S3_BUCKET="aws-gpu-monitoring-logs"
S3_PATH="spot-monitoring/logs"
AWS_REGIONS="us-east-1 us-east-2 us-west-1 us-west-2 eu-west-1 eu-west-2 eu-central-1 ap-southeast-1 ap-southeast-2 ap-northeast-1 ap-northeast-2"
INSTANCE_TYPES="g4dn.xlarge g4dn.2xlarge"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Start the log file with timestamp and header
echo "AWS Spot Instance Price Monitor - Run at: $(date)" > ${LOG_FILE}
echo "=======================================================" >> ${LOG_FILE}

# Function to query spot prices for a specific region and instance type
get_spot_prices() {
    local region=$1
    local instance=$2
    
    echo "Region: ${region} | Instance Type: ${instance}" >> ${LOG_FILE}
    echo "---------------------------------------------------------" >> ${LOG_FILE}
    
    # Get current spot price
    aws ec2 describe-spot-price-history \
        --region ${region} \
        --instance-types ${instance} \
        --product-descriptions "Linux/UNIX" \
        --start-time $(date -u +"%Y-%m-%dT%H:%M:%S") \
        --end-time $(date -u +"%Y-%m-%dT%H:%M:%S") \
        --query 'SpotPriceHistory[*].{AvailabilityZone:AvailabilityZone, Price:SpotPrice}' \
        --output table >> ${LOG_FILE} 2>&1
    
    echo "" >> ${LOG_FILE}
}

# Loop through regions and instance types
for region in ${AWS_REGIONS}; do
    for instance in ${INSTANCE_TYPES}; do
        get_spot_prices ${region} ${instance}
    done
    echo "=======================================================" >> ${LOG_FILE}
done

# Add daily and hourly cost estimates
echo "ESTIMATED HOURLY AND DAILY COST (Assuming 100% utilization)" >> ${LOG_FILE}
echo "=======================================================" >> ${LOG_FILE}

for region in ${AWS_REGIONS}; do
    echo "Region: ${region}" >> ${LOG_FILE}
    for instance in ${INSTANCE_TYPES}; do
        # Get the lowest price in the region for the instance type
        lowest_price=$(aws ec2 describe-spot-price-history \
            --region ${region} \
            --instance-types ${instance} \
            --product-descriptions "Linux/UNIX" \
            --start-time $(date -u +"%Y-%m-%dT%H:%M:%S") \
            --end-time $(date -u +"%Y-%m-%dT%H:%M:%S") \
            --query 'SpotPriceHistory[*].SpotPrice' \
            --output text | sort -n | head -1)
        
        if [ -n "$lowest_price" ]; then
            hourly_cost=$(echo "$lowest_price" | awk '{print $1}')
            daily_cost=$(echo "$hourly_cost * 24" | bc -l)
            
            echo "${instance}: $hourly_cost/hour (~$daily_cost/day)" >> ${LOG_FILE}
        else
            echo "${instance}: No price data available" >> ${LOG_FILE}
        fi
    done
    echo "---------------------------------------------------------" >> ${LOG_FILE}
done

# Copy to latest log file
cp ${LOG_FILE} ${LATEST_LOG}

# Upload log file to S3
aws s3 cp ${LOG_FILE} s3://${S3_BUCKET}/${S3_PATH}/spot_pricing_${TIMESTAMP}.log
aws s3 cp ${LATEST_LOG} s3://${S3_BUCKET}/${S3_PATH}/spot_pricing_latest.log

echo "Log uploaded to s3://${S3_BUCKET}/${S3_PATH}/spot_pricing_${TIMESTAMP}.log"

# Clean up old log files (keep only last 100)
find ${LOG_DIR} -name "spot_pricing_*.log" | sort -r | tail -n +100 | xargs -r rm

exit 0

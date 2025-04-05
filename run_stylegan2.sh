#!/bin/bash
# Create necessary directories
mkdir -p ./models ./data ./samples ./output ./logs

# Log start time of training
TRAIN_START=$(date +%s)
echo "Training started at: $(date)"

# Run training script
python3 training_workload.py
TRAIN_EXIT_CODE=$?

# Log end time of training
TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
echo "Training completed at: $(date)"
echo "Training duration: $TRAIN_DURATION seconds"

# Check if training failed
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "Error: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

# Find the latest model file
LATEST_MODEL=$(find ./output -name "*.pkl" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$LATEST_MODEL" ]; then
    echo "Error: No model file found. Training may not have completed successfully."
    exit 1
fi

echo "Found model file: $LATEST_MODEL"

# Log start time of inference and evaluation
INFER_START=$(date +%s)
echo "Inference and evaluation started at: $(date)"

# Run inference script with various options
echo "Generating samples..."
python3 inference_workload.py --network="$LATEST_MODEL" --seeds=0-63 --outdir=./output/generated_images

# Log end time of inference
INFER_END=$(date +%s)
INFER_DURATION=$((INFER_END - INFER_START))
echo "Inference and evaluation completed at: $(date)"
echo "Inference and evaluation duration: $INFER_DURATION seconds"

# Summary
echo "----------------------------------------"
echo "StyleGAN2-ADA Training and Evaluation Summary:"
echo "----------------------------------------"
echo "Training duration: $TRAIN_DURATION seconds"
echo "Inference duration: $INFER_DURATION seconds"
echo "Total execution time: $((TRAIN_DURATION + INFER_DURATION)) seconds"
echo "All outputs saved to: ./output"
echo "Model file: $LATEST_MODEL"
echo "----------------------------------------"

exit 0

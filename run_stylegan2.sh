#!/bin/bash

# Create necessary directories
mkdir -p ./models ./data ./samples ./output ./logs

# Log start time of training
TRAIN_START=$(date +%s)
echo "Training started at: $(date)"

# Run training script
python3 training_workload.py

# Log end time of training
TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
echo "Training completed at: $(date)"
echo "Training duration: $TRAIN_DURATION seconds"

# Check if training was successful by looking for final model
if [ ! -f "./models/final_model.pth" ]; then
    echo "Error: Training did not produce a final model file."
    echo "Check training logs for errors."
    exit 1
fi

# Log start time of inference and evaluation
INFER_START=$(date +%s)
echo "Inference and evaluation started at: $(date)"

# Run inference script with various options
echo "Generating samples..."
python3 dcgan_inference.py --checkpoint ./models/final_model.pth --num_samples 64 --output_dir ./output

echo "Generating latent space interpolation..."
python3 dcgan_inference.py --checkpoint ./models/final_model.pth --interpolate --output_dir ./output

echo "Evaluating model..."
python3 dcgan_inference.py --checkpoint ./models/final_model.pth --evaluate --output_dir ./output

# Log end time of inference
INFER_END=$(date +%s)
INFER_DURATION=$((INFER_END - INFER_START))
echo "Inference and evaluation completed at: $(date)"
echo "Inference and evaluation duration: $INFER_DURATION seconds"

# Summary
echo "----------------------------------------"
echo "DCGAN Training and Evaluation Summary:"
echo "----------------------------------------"
echo "Training duration: $TRAIN_DURATION seconds"
echo "Inference duration: $INFER_DURATION seconds"
echo "Total execution time: $((TRAIN_DURATION + INFER_DURATION)) seconds"
echo "All outputs saved to: ./output"
echo "Model checkpoints saved to: ./models"
echo "----------------------------------------"

# Check if any FID results were generated
if [ -f "./output/evaluation_results.txt" ]; then
    echo "Evaluation Results:"
    cat ./output/evaluation_results.txt
    echo "----------------------------------------"
fi

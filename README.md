# StyleGAN2-ADA Workload Modules

This directory contains modular Python scripts for working with StyleGAN2-ADA across different industry datasets. Each module is designed to be flexible, allowing easy dataset swapping without modifying core code logic.

## Workload Categories

### 1. Training Workload

The `training_workload.py` script provides a flexible interface for training StyleGAN2-ADA models on custom datasets:

```bash
python training_workload.py --dataset-path=./datasets/automotive --image-size=256 --outdir=./results
```

### 2. Inference Workload

The `inference_workload.py` script allows generating images with trained models:

```bash
python inference_workload.py --network=auto_model.pkl --seeds=0-10 --outdir=out --industry=automotive
```

### 3. Optimization Workloads

Three specialized optimization scripts are provided:

#### a. Fine-Tuning Optimization

```bash
python fine_tuning_optimization.py --dataset-path=./datasets/medical --resume=ffhq256 --outdir=./results
```

#### b. Latent Vector Optimization

```bash
python latent_vector_optimization.py --network=fashion_model.pkl --target=./targets/fashion --outdir=./results
```

#### c. Mixed Precision Training Optimization

```bash
python mixed_precision_optimization.py --dataset-path=./datasets/automotive --mixed-precision-mode=aggressive --outdir=./results
```

## Automated Sequential Workflow

A Bash script is provided to run all four workloads sequentially with automatic S3 uploading of results:

```bash
./stylegan-script.sh
```

This script:
1. Runs basic training and inference
2. Uses the basic model for fine-tuning
3. Performs latent vector optimization
4. Runs mixed precision training
5. Uploads all logs, models, and results to S3 after each workflow completes

## Continuous Operation Setup

For running StyleGAN2 workflows continuously with monitoring:

```bash
# Install as a system service
sudo ./install.sh

# Start continuous operation manually
./continuous-stylegan.sh
```

The continuous operation:
- Runs complete StyleGAN2 workflow cycles indefinitely
- Automatically uploads results to S3 after each cycle
- Integrates with Docker monitoring stack
- Provides monitoring via Prometheus and Grafana

## Docker Monitoring Integration

The included Docker Compose setup provides real-time monitoring of all StyleGAN2 workflows:

```bash
# Start monitoring stack
docker-compose up -d

# Run StyleGAN2 with monitoring 
./run-docker-stylegan.sh
```

The monitoring stack includes:
- Prometheus for metrics collection
- Grafana for visualization
- NVIDIA DCGM exporter for GPU metrics
- Node exporter for system metrics
- CloudWatch exporter for AWS metrics
- Thanos for long-term storage of metrics

## Dataset Structure

All scripts support swapping datasets without modifying code. Datasets should follow this structure:

```
dataset_directory/
├── 000000.png
├── 000001.png
├── ...
└── dataset.json (optional, for labels)
```

Images should be in a common format (PNG, JPG, etc.) and ideally pre-processed to the target resolution.

### Dataset JSON Format (Optional)

For conditional generation, include a `dataset.json` file:

```json
{
  "labels": [
    ["000000.png", [0, 1, 0, ...]],
    ["000001.png", [1, 0, 0, ...]],
    ...
  ]
}
```

## S3 Integration

All scripts support automatic uploading of logs, models, and generated images to AWS S3:

```bash
# Configure AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export AWS_DEFAULT_REGION=us-east-1
```

S3 data is organized by:
- Date and timestamp
- Workflow type (basic_training, finetuned, etc.)
- Data type (logs, models, images)

## Industry-Specific Usage

### Healthcare

```bash
# Train on medical images
python training_workload.py --dataset-path=./datasets/medical --image-size=256 --outdir=./results

# Fine-tune an existing model
python fine_tuning_optimization.py --dataset-path=./datasets/medical --resume=ffhq256 --outdir=./results --freezed=6

# Optimize latent vectors for specific medical images
python latent_vector_optimization.py --network=medical_model.pkl --target=./targets/medical --outdir=./results --industry=healthcare
```

### Automotive

```bash
# Train on automotive images
python training_workload.py --dataset-path=./datasets/automotive --image-size=512 --outdir=./results

# Generate automotive-style images
python inference_workload.py --network=auto_model.pkl --seeds=0-50 --outdir=out --industry=automotive

# Use mixed precision for faster training
python mixed_precision_optimization.py --dataset-path=./datasets/automotive --mixed-precision-mode=aggressive --outdir=./results
```

### Fashion

```bash
# Train on fashion images
python training_workload.py --dataset-path=./datasets/fashion --image-size=1024 --outdir=./results

# Generate fashion images with specific style
python inference_workload.py --network=fashion_model.pkl --seeds=100-150 --outdir=out --industry=fashion

# Optimize latent vectors for specific fashion designs
python latent_vector_optimization.py --network=fashion_model.pkl --target=./targets/fashion --outdir=./results --latent-space=w+
```

## Fashion-MNIST Example

A complete example workflow using Fashion-MNIST is included:

```bash
# Prepare Fashion-MNIST dataset
python prepare_fashion_mnist.py --output-dir=./datasets/fashion-mnist --image-size=64

# Run the complete sequential workflow
./stylegan-script.sh
```

This will run all four workloads on the Fashion-MNIST dataset and generate an HTML gallery with comparison visualizations.

## Configuration Files

All scripts support YAML or JSON configuration files for easier management:

```bash
python training_workload.py --dataset-path=./datasets/medical --config=./configs/medical.yaml --outdir=./results
```

Example configuration (medical.yaml):

```yaml
dataset:
  path: ./datasets/medical
  image_size: 256
  use_labels: true
  mirror: true
gpus: 2
snap: 100
kimg: 10000
metrics: ['fid50k_full', 'kid50k_full']
aug: ada
```

## Dependencies

- PyTorch (>= 1.7.0)
- Python (>= 3.6)
- CUDA (>= 11.0)
- Docker and Docker Compose (for monitoring)
- AWS CLI (for S3 uploads)
- Additional requirements match the base StyleGAN2-ADA repository

## Monitoring Dashboard

Access the monitoring dashboard at http://localhost:3000 with:
- Username: admin
- Password: admin

The dashboard provides real-time metrics on:
- GPU utilization
- Memory usage
- Training progress
- Model generation metrics
- System resource utilization

## Notes

- For multi-GPU training, ensure CUDA is properly configured
- Mixed precision training works best on NVIDIA GPUs with Tensor Cores (Volta, Turing, or Ampere architectures)
- When using the latent vector optimization, LPIPS dependencies will be installed if not present
- For continuous operation, ensure sufficient disk space for logs and models
- All S3 uploads include instance metadata for tracking computational resources

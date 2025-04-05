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
- Additional requirements match the base StyleGAN2-ADA repository

## Notes

- For multi-GPU training, ensure CUDA is properly configured
- Mixed precision training works best on NVIDIA GPUs with Tensor Cores (Volta, Turing, or Ampere architectures)
- When using the latent vector optimization, LPIPS dependencies will be installed if not present
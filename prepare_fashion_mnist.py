#!/usr/bin/env python3
"""
Fashion-MNIST Dataset Preparation Script for StyleGAN2-ADA

This script downloads the Fashion-MNIST dataset and converts it to 
a format suitable for StyleGAN2-ADA training.

The Fashion-MNIST dataset consists of 60,000 training images and 10,000 test
images, each of size 28x28 pixels. This script will:
1. Download the dataset
2. Resize images to 64x64 (or other specified size)
3. Save images in the StyleGAN2-ADA expected format
4. Create a dataset.json file with labels

Note: StyleGAN2-ADA works best with higher resolution images, but we're
upsampling Fashion-MNIST as an example.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Fashion-MNIST for StyleGAN2-ADA')
    parser.add_argument('--output-dir', type=str, default='./datasets/fashion-mnist',
                        help='Output directory for dataset')
    parser.add_argument('--image-size', type=int, default=64,
                        help='Size to resize images to (default: 64)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to use (default: all)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading Fashion-MNIST dataset...")
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
    ])
    
    # Download dataset
    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    # Limit samples if specified
    if args.samples is not None:
        indices = torch.randperm(len(train_dataset))[:args.samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    # Class names for reference
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Prepare dataset.json
    labels_data = {"labels": []}
    
    # Convert and save images
    print(f"Processing {len(train_dataset)} images...")
    for i, (img_tensor, label) in enumerate(tqdm(train_dataset)):
        # Convert to numpy array and scale to [0, 255]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Convert grayscale to RGB
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        
        # Create PIL image
        img = Image.fromarray(img_np)
        
        # Create filename
        filename = f"{i:06d}.png"
        filepath = os.path.join(args.output_dir, filename)
        
        # Save image
        img.save(filepath)
        
        # Create one-hot encoded label
        one_hot = [0] * len(class_names)
        one_hot[label] = 1
        
        # Add to labels data
        labels_data["labels"].append([filename, one_hot])
    
    # Save dataset.json
    json_path = os.path.join(args.output_dir, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(labels_data, f)
    
    print(f"Dataset prepared successfully:")
    print(f"- {len(train_dataset)} images saved to {args.output_dir}")
    print(f"- Images resized to {args.image_size}x{args.image_size}")
    print(f"- Labels saved to {json_path}")
    print("\nClass distribution:")
    for i, name in enumerate(class_names):
        count = sum(1 for _, label in labels_data["labels"] if label[i] == 1)
        print(f"- {name}: {count} images")

if __name__ == "__main__":
    main()
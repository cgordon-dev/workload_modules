# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Inference workload for StyleGAN2-ADA with easy dataset swapping"""

import os
import re
import json
import yaml
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

def load_config_file(config_file):
    """Load configuration from a YAML or JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        config: Parsed configuration dictionary
    """
    if config_file.endswith('.json'):
        with open(config_file, 'r') as f:
            return json.load(f)
    elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_file}")

#----------------------------------------------------------------------------

def generate_images(G, device, seeds, truncation_psi, noise_mode, outdir, class_idx=None, projected_w=None, industry=None):
    """Generate images using pretrained generator network.
    
    Args:
        G: Generator network
        device: Torch device to run on
        seeds: List of random seeds
        truncation_psi: Truncation parameter
        noise_mode: Noise mode for synthesis ('const', 'random', 'none')
        outdir: Output directory
        class_idx: Class index for conditional generation
        projected_w: Projected W latent vectors
        industry: Optional industry name for file organization
    
    Returns:
        List of generated image paths
    """
    os.makedirs(outdir, exist_ok=True)
    if industry is not None:
        industry_dir = os.path.join(outdir, industry)
        os.makedirs(industry_dir, exist_ok=True)
        outdir = industry_dir
    
    generated_images = []

    # Synthesize the result of a W projection.
    if projected_w is not None:
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_path = f'{outdir}/proj{idx:02d}.png'
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(img_path)
            generated_images.append(img_path)
        return generated_images

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise ValueError('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('Warning: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print(f'Generating image for seed {seed} ({seed_idx+1}/{len(seeds)}) ...')
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_path = f'{outdir}/seed{seed:04d}.png'
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(img_path)
        generated_images.append(img_path)
    
    return generated_images

#----------------------------------------------------------------------------

def process_industry_dataset(industry_name: str, dataset_path: str, seeds: List[int], network_pkl: str, truncation_psi: float, 
                              noise_mode: str, outdir: str, class_idx: Optional[int] = None, projected_w: Optional[str] = None) -> Dict[str, Any]:
    """Process dataset for a specific industry.
    
    Args:
        industry_name: Name of the industry (e.g., healthcare, automotive)
        dataset_path: Path to the industry-specific dataset
        seeds: List of random seeds for generation
        network_pkl: Path or URL to the network pickle
        truncation_psi: Truncation parameter
        noise_mode: Noise mode for synthesis
        outdir: Output directory
        class_idx: Class index for conditional generation
        projected_w: Projected W latent vectors
        
    Returns:
        Results dictionary with industry metrics
    """
    print(f'Processing {industry_name} dataset from {dataset_path}')
    
    # Prepare input/output directories
    os.makedirs(outdir, exist_ok=True)
    industry_dir = os.path.join(outdir, industry_name)
    os.makedirs(industry_dir, exist_ok=True)
    
    # Load network
    print(f'Loading network from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        
    # Generate images
    images = generate_images(
        G=G,
        device=device,
        seeds=seeds,
        truncation_psi=truncation_psi,
        noise_mode=noise_mode,
        outdir=industry_dir,
        class_idx=class_idx,
        projected_w=projected_w,
        industry=industry_name
    )
    
    results = {
        'industry': industry_name,
        'dataset_path': dataset_path,
        'generated_images': images,
        'network_pkl': network_pkl,
        'truncation_psi': truncation_psi
    }
    
    # Save results summary
    with open(os.path.join(industry_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True)
@click.option('--seeds', type=num_range, help='List of random seeds', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1.0, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--industry', help='Industry name for organization', type=str)
@click.option('--config', help='Path to configuration file (yaml or json)', type=str)
@click.option('--batch-size', help='Batch size for generation', type=int, default=1)
def main(network_pkl, outdir, seeds, truncation_psi, class_idx, noise_mode, projected_w, industry, config, batch_size):
    """Generate images using a pretrained StyleGAN2-ADA network.
    
    Examples:
    
    # Generate automotive images
    python inference_workload.py --network=auto_model.pkl --seeds=0-10 --outdir=out --industry=automotive
    
    # Generate healthcare images with custom configuration
    python inference_workload.py --network=medical_model.pkl --config=medical_config.yaml --outdir=out
    
    # Generate images from a projected latent space
    python inference_workload.py --network=fashion_model.pkl --projected-w=projected_w.npz --outdir=out --industry=fashion
    """
    
    # Handle configuration file if provided
    if config is not None:
        cfg = load_config_file(config)
        # Override with config values, but command-line arguments take precedence
        network_pkl = network_pkl or cfg.get('network')
        outdir = outdir or cfg.get('outdir')
        seeds = seeds or num_range(cfg.get('seeds', '0-10'))
        truncation_psi = truncation_psi if truncation_psi != 1.0 else cfg.get('truncation_psi', 1.0)
        class_idx = class_idx or cfg.get('class_idx')
        noise_mode = noise_mode if noise_mode != 'const' else cfg.get('noise_mode', 'const')
        projected_w = projected_w or cfg.get('projected_w')
        industry = industry or cfg.get('industry')
        batch_size = batch_size if batch_size != 1 else cfg.get('batch_size', 1)
        
        # Handle industry-specific datasets in config
        if 'industries' in cfg and not industry:
            results = {}
            for ind_name, ind_config in cfg['industries'].items():
                ind_results = process_industry_dataset(
                    industry_name=ind_name,
                    dataset_path=ind_config.get('dataset_path', ''),
                    seeds=num_range(ind_config.get('seeds', '0-10')),
                    network_pkl=ind_config.get('network', network_pkl),
                    truncation_psi=ind_config.get('truncation_psi', truncation_psi),
                    noise_mode=ind_config.get('noise_mode', noise_mode),
                    outdir=outdir,
                    class_idx=ind_config.get('class_idx', class_idx),
                    projected_w=ind_config.get('projected_w', projected_w)
                )
                results[ind_name] = ind_results
            
            # Save overall results
            with open(os.path.join(outdir, 'all_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            return
    
    # Process single industry or default case
    print(f'Loading network from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    generate_images(
        G=G,
        device=device,
        seeds=seeds,
        truncation_psi=truncation_psi,
        noise_mode=noise_mode,
        outdir=outdir,
        class_idx=class_idx,
        projected_w=projected_w,
        industry=industry
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
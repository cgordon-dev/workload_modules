# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Latent vector optimization workload for StyleGAN2-ADA with industry-specific configurations"""

import os
import re
import json
import yaml
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import lpips

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path

import legacy
from training.networks import Generator

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

def load_target_images(target_path: str, resolution: int, device) -> torch.Tensor:
    """Load and preprocess target images.
    
    Args:
        target_path: Path to target image or directory
        resolution: Target resolution for images
        device: PyTorch device
        
    Returns:
        target_images: Tensor of preprocessed target images
    """
    if os.path.isdir(target_path):
        # Load all images from a directory
        target_images = []
        for filename in os.listdir(target_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                img_path = os.path.join(target_path, filename)
                img = PIL.Image.open(img_path).convert('RGB')
                img = img.resize((resolution, resolution), PIL.Image.LANCZOS)
                arr = np.array(img, dtype=np.float32)
                arr = arr.transpose(2, 0, 1) / 255.0 * 2.0 - 1.0  # HWC => CHW, [0,255] => [-1,1]
                target_images.append(torch.from_numpy(arr))
        
        if not target_images:
            raise ValueError(f"No valid images found in {target_path}")
        
        target_images = torch.stack(target_images).to(device)
    else:
        # Load a single image
        img = PIL.Image.open(target_path).convert('RGB')
        img = img.resize((resolution, resolution), PIL.Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        arr = arr.transpose(2, 0, 1) / 255.0 * 2.0 - 1.0  # HWC => CHW, [0,255] => [-1,1]
        target_images = torch.from_numpy(arr).unsqueeze(0).to(device)
    
    return target_images

#----------------------------------------------------------------------------

def get_perceptual_loss(perception_type='lpips'):
    """Create perceptual loss function.
    
    Args:
        perception_type: Type of perceptual loss ('lpips', 'vgg', or 'none')
        
    Returns:
        Function that calculates perceptual loss
    """
    if perception_type.lower() == 'none':
        # Use simple L2 loss
        def perception_loss(img1, img2):
            return F.mse_loss(img1, img2)
        
    elif perception_type.lower() == 'lpips':
        # Use LPIPS perceptual loss
        lpips_model = lpips.LPIPS(net='vgg').eval()
        
        def perception_loss(img1, img2):
            # LPIPS expects inputs in [-1, 1] range
            return lpips_model(img1, img2).mean()
        
    else:
        raise ValueError(f"Unsupported perception_type: {perception_type}")
    
    return perception_loss

#----------------------------------------------------------------------------

def optimize_latent_vectors(
    G: Generator,
    target_images: torch.Tensor,
    device: torch.device,
    num_steps: int = 1000,
    initial_learning_rate: float = 0.1,
    initial_latents: Optional[torch.Tensor] = None,
    latent_space: str = 'w',
    truncation_psi: float = 0.7,
    noise_mode: str = 'const',
    perceptual_loss_type: str = 'lpips',
    verbose: bool = True
) -> Tuple[torch.Tensor, List[float]]:
    """Optimize latent vectors to match target images.
    
    Args:
        G: StyleGAN2 generator
        target_images: Target images tensor [N, C, H, W]
        device: PyTorch device
        num_steps: Number of optimization steps
        initial_learning_rate: Starting learning rate
        initial_latents: Optional initial latent vectors
        latent_space: Latent space to optimize in ('z', 'w', 'w+')
        truncation_psi: Truncation psi for generation
        noise_mode: Noise mode for synthesis
        perceptual_loss_type: Type of perceptual loss
        verbose: Whether to show progress bar
        
    Returns:
        Tuple of optimized latent vectors and loss history
    """
    assert latent_space in ['z', 'w', 'w+'], f"Unsupported latent space: {latent_space}"
    
    batch_size = target_images.shape[0]
    perception_loss = get_perceptual_loss(perceptual_loss_type)
    
    # Initialize latent vectors
    if initial_latents is None:
        if latent_space == 'z':
            latents = torch.randn((batch_size, G.z_dim), device=device, requires_grad=True)
        elif latent_space == 'w':
            z = torch.randn((batch_size, G.z_dim), device=device)
            with torch.no_grad():
                w = G.mapping(z, None, truncation_psi=truncation_psi)
            latents = w.detach().clone().requires_grad_(True)
        else:  # w+
            z = torch.randn((batch_size, G.z_dim), device=device)
            with torch.no_grad():
                w = G.mapping(z, None, truncation_psi=truncation_psi)
                w_plus = w.unsqueeze(1).repeat(1, G.num_ws, 1)
            latents = w_plus.detach().clone().requires_grad_(True)
    else:
        latents = initial_latents.clone().to(device).requires_grad_(True)
    
    # Use Adam optimizer with learning rate schedule
    optimizer = torch.optim.Adam([latents], lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    loss_history = []
    iterator = range(num_steps)
    if verbose:
        iterator = tqdm(iterator, desc="Optimizing latent vectors")
    
    for step in iterator:
        optimizer.zero_grad()
        
        # Generate images from current latents
        if latent_space == 'z':
            with torch.no_grad():
                w = G.mapping(latents, None, truncation_psi=truncation_psi)
            generated_images = G.synthesis(w, noise_mode=noise_mode)
        elif latent_space == 'w':
            generated_images = G.synthesis(latents.unsqueeze(1).repeat(1, G.num_ws, 1), noise_mode=noise_mode)
        else:  # w+
            generated_images = G.synthesis(latents, noise_mode=noise_mode)
        
        # Calculate loss
        loss = perception_loss(generated_images, target_images)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update learning rate every 100 steps
        if (step + 1) % 100 == 0:
            scheduler.step()
        
        loss_history.append(loss.item())
        
        if verbose and (step + 1) % 100 == 0:
            tqdm.write(f"Step {step + 1}/{num_steps}, Loss: {loss.item():.6f}")
    
    return latents.detach(), loss_history

#----------------------------------------------------------------------------

def save_optimized_images(
    G: Generator,
    latents: torch.Tensor,
    outdir: str,
    latent_space: str = 'w',
    truncation_psi: float = 0.7,
    noise_mode: str = 'const',
    suffix: str = ''
) -> List[str]:
    """Save images generated from optimized latent vectors.
    
    Args:
        G: StyleGAN2 generator
        latents: Optimized latent vectors
        outdir: Output directory
        latent_space: Latent space used ('z', 'w', 'w+')
        truncation_psi: Truncation psi for generation
        noise_mode: Noise mode for synthesis
        suffix: Optional suffix for filenames
        
    Returns:
        List of saved image paths
    """
    os.makedirs(outdir, exist_ok=True)
    image_paths = []
    
    # Generate images from latent vectors
    with torch.no_grad():
        if latent_space == 'z':
            w = G.mapping(latents, None, truncation_psi=truncation_psi)
            generated_images = G.synthesis(w, noise_mode=noise_mode)
        elif latent_space == 'w':
            generated_images = G.synthesis(latents.unsqueeze(1).repeat(1, G.num_ws, 1), noise_mode=noise_mode)
        else:  # w+
            generated_images = G.synthesis(latents, noise_mode=noise_mode)
    
    # Convert and save each image
    for i, img in enumerate(generated_images):
        img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        img_path = os.path.join(outdir, f'optimized{i:04d}{suffix}.png')
        PIL.Image.fromarray(img, 'RGB').save(img_path)
        image_paths.append(img_path)
    
    # Save latent vectors for future use
    np_latents = latents.cpu().numpy()
    if latent_space == 'w+':
        np.save(os.path.join(outdir, f'latents_wplus{suffix}.npy'), np_latents)
    elif latent_space == 'w':
        np.save(os.path.join(outdir, f'latents_w{suffix}.npy'), np_latents)
    else:
        np.save(os.path.join(outdir, f'latents_z{suffix}.npy'), np_latents)
    
    # Save in projected_w format for compatibility with other scripts
    if latent_space in ['w', 'w+']:
        w_latents = latents if latent_space == 'w+' else latents.unsqueeze(1).repeat(1, G.num_ws, 1)
        np.savez(os.path.join(outdir, f'projected_w{suffix}.npz'), w=w_latents.cpu().numpy())
    
    return image_paths

#----------------------------------------------------------------------------

def process_industry_dataset(
    industry_name: str,
    target_path: str,
    network_pkl: str,
    outdir: str,
    num_steps: int = 1000,
    latent_space: str = 'w+',
    perceptual_loss_type: str = 'lpips',
    truncation_psi: float = 0.7,
    noise_mode: str = 'const',
    seed: int = 0
) -> Dict[str, Any]:
    """Process dataset for a specific industry.
    
    Args:
        industry_name: Name of the industry (e.g., healthcare, automotive)
        target_path: Path to target images
        network_pkl: Path or URL to the network pickle
        outdir: Output directory
        num_steps: Number of optimization steps
        latent_space: Latent space to optimize in
        perceptual_loss_type: Type of perceptual loss
        truncation_psi: Truncation psi for generation
        noise_mode: Noise mode for synthesis
        seed: Random seed
        
    Returns:
        Results dictionary with industry metrics
    """
    print(f'Processing {industry_name} dataset from {target_path}')
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Prepare input/output directories
    os.makedirs(outdir, exist_ok=True)
    industry_dir = os.path.join(outdir, industry_name)
    os.makedirs(industry_dir, exist_ok=True)
    
    # Load network
    print(f'Loading network from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    # Load target images
    resolution = G.img_resolution
    print(f'Loading target images at resolution {resolution}x{resolution}...')
    target_images = load_target_images(target_path, resolution, device)
    print(f'Loaded {len(target_images)} target images')
    
    # Optimize latent vectors
    print(f'Optimizing {len(target_images)} latent vectors in {latent_space} space using {perceptual_loss_type} perceptual loss...')
    latents, loss_history = optimize_latent_vectors(
        G=G,
        target_images=target_images,
        device=device,
        num_steps=num_steps,
        latent_space=latent_space,
        truncation_psi=truncation_psi,
        noise_mode=noise_mode,
        perceptual_loss_type=perceptual_loss_type,
        verbose=True
    )
    
    # Save results
    print('Saving optimized images and latent vectors...')
    image_paths = save_optimized_images(
        G=G,
        latents=latents,
        outdir=industry_dir,
        latent_space=latent_space,
        truncation_psi=truncation_psi,
        noise_mode=noise_mode,
        suffix=f'_{industry_name}'
    )
    
    # Save loss history
    np.save(os.path.join(industry_dir, f'loss_history_{industry_name}.npy'), np.array(loss_history))
    
    # Create and save results summary
    results = {
        'industry': industry_name,
        'target_path': target_path,
        'generated_images': image_paths,
        'network_pkl': network_pkl,
        'num_steps': num_steps,
        'latent_space': latent_space,
        'perceptual_loss_type': perceptual_loss_type,
        'truncation_psi': truncation_psi,
        'final_loss': loss_history[-1] if loss_history else None
    }
    
    with open(os.path.join(industry_dir, 'optimization_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_path', help='Target image or directory', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True)
@click.option('--num-steps', help='Number of optimization steps', type=int, default=1000)
@click.option('--latent-space', help='Latent space to optimize in', type=click.Choice(['z', 'w', 'w+']), default='w+')
@click.option('--perceptual-loss', help='Perceptual loss type', type=click.Choice(['lpips', 'none']), default='lpips')
@click.option('--initial-lr', help='Initial learning rate', type=float, default=0.1)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--seed', help='Random seed', type=int, default=0)
@click.option('--industry', help='Industry name for organization', type=str)
@click.option('--config', help='Path to configuration file (yaml or json)', type=str)
def main(
    network_pkl: str,
    target_path: str,
    outdir: str,
    num_steps: int,
    latent_space: str,
    perceptual_loss: str,
    initial_lr: float,
    truncation_psi: float,
    noise_mode: str,
    seed: int,
    industry: Optional[str],
    config: Optional[str]
):
    """Optimize latent vectors to match target images.
    
    Examples:
    
    # Optimize latent vectors for automotive images
    python latent_vector_optimization.py --network=auto_model.pkl --target=./targets/cars --outdir=./results --industry=automotive
    
    # Optimize latent vectors for medical images with custom config
    python latent_vector_optimization.py --network=medical_model.pkl --config=medical_config.yaml --outdir=./results
    
    # Optimize latent vectors in Z space
    python latent_vector_optimization.py --network=fashion_model.pkl --target=./targets/fashion --outdir=./results --latent-space=z
    """
    
    # Handle configuration file if provided
    if config is not None:
        cfg = load_config_file(config)
        # Override with config values, but command-line arguments take precedence
        network_pkl = network_pkl or cfg.get('network')
        outdir = outdir or cfg.get('outdir')
        target_path = target_path or cfg.get('target')
        num_steps = num_steps if num_steps != 1000 else cfg.get('num_steps', 1000)
        latent_space = latent_space if latent_space != 'w+' else cfg.get('latent_space', 'w+')
        perceptual_loss = perceptual_loss if perceptual_loss != 'lpips' else cfg.get('perceptual_loss', 'lpips')
        initial_lr = initial_lr if initial_lr != 0.1 else cfg.get('initial_lr', 0.1)
        truncation_psi = truncation_psi if truncation_psi != 0.7 else cfg.get('truncation_psi', 0.7)
        noise_mode = noise_mode if noise_mode != 'const' else cfg.get('noise_mode', 'const')
        seed = seed if seed != 0 else cfg.get('seed', 0)
        industry = industry or cfg.get('industry')
        
        # Handle industry-specific settings in config
        if 'industries' in cfg and not industry:
            results = {}
            for ind_name, ind_config in cfg['industries'].items():
                ind_results = process_industry_dataset(
                    industry_name=ind_name,
                    target_path=ind_config.get('target_path', target_path),
                    network_pkl=ind_config.get('network', network_pkl),
                    outdir=outdir,
                    num_steps=ind_config.get('num_steps', num_steps),
                    latent_space=ind_config.get('latent_space', latent_space),
                    perceptual_loss_type=ind_config.get('perceptual_loss', perceptual_loss),
                    truncation_psi=ind_config.get('truncation_psi', truncation_psi),
                    noise_mode=ind_config.get('noise_mode', noise_mode),
                    seed=ind_config.get('seed', seed)
                )
                results[ind_name] = ind_results
            
            # Save overall results
            with open(os.path.join(outdir, 'all_optimization_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            return
    
    # Process single industry or default case
    if industry:
        process_industry_dataset(
            industry_name=industry,
            target_path=target_path,
            network_pkl=network_pkl,
            outdir=outdir,
            num_steps=num_steps,
            latent_space=latent_space,
            perceptual_loss_type=perceptual_loss,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
            seed=seed
        )
    else:
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load network
        print(f'Loading network from "{network_pkl}"...')
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)
        
        # Load target images
        resolution = G.img_resolution
        print(f'Loading target images at resolution {resolution}x{resolution}...')
        target_images = load_target_images(target_path, resolution, device)
        print(f'Loaded {len(target_images)} target images')
        
        # Optimize latent vectors
        print(f'Optimizing {len(target_images)} latent vectors in {latent_space} space using {perceptual_loss} perceptual loss...')
        latents, loss_history = optimize_latent_vectors(
            G=G,
            target_images=target_images,
            device=device,
            num_steps=num_steps,
            initial_learning_rate=initial_lr,
            latent_space=latent_space,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
            perceptual_loss_type=perceptual_loss,
            verbose=True
        )
        
        # Save results
        print('Saving optimized images and latent vectors...')
        save_optimized_images(
            G=G,
            latents=latents,
            outdir=outdir,
            latent_space=latent_space,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode
        )
        
        # Save loss history
        np.save(os.path.join(outdir, 'loss_history.npy'), np.array(loss_history))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
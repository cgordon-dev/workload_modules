# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Training workload for StyleGAN2-ADA with easy dataset swapping"""

import os
import click
import re
import json
import tempfile
import torch
import dnnlib
import yaml
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path

from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def load_dataset(dataset_path, image_size=None, use_labels=True, max_size=None, xflip=False):
    """Load dataset with configurable parameters.
    
    Args:
        dataset_path: Path to the dataset directory or zip file
        image_size: Required resolution for images (height=width)
        use_labels: Whether to use labels from dataset.json
        max_size: Maximum number of images to use
        xflip: Whether to augment with horizontal flips
        
    Returns:
        training_set_kwargs: Dataset configuration dict for training
    """
    training_set_kwargs = dnnlib.EasyDict()
    training_set_kwargs.class_name = 'training.dataset.ImageFolderDataset'
    training_set_kwargs.path = dataset_path
    training_set_kwargs.use_labels = use_labels
    training_set_kwargs.max_size = max_size
    training_set_kwargs.xflip = xflip
    if image_size is not None:
        training_set_kwargs.resolution = image_size
    
    return training_set_kwargs

#----------------------------------------------------------------------------

def setup_training_config(config):
    """Set up training configuration based on parsed arguments or config file.
    
    Args:
        config: Configuration dictionary with training parameters
        
    Returns:
        args: Training arguments for StyleGAN2-ADA
    """
    args = dnnlib.EasyDict()
    
    # Dataset options
    dataset_config = config.get('dataset', {})
    args.training_set_kwargs = load_dataset(
        dataset_path=dataset_config.get('path'),
        image_size=dataset_config.get('image_size'),
        use_labels=dataset_config.get('use_labels', True),
        max_size=dataset_config.get('max_size'),
        xflip=dataset_config.get('mirror', False)
    )
    
    # General options
    args.num_gpus = config.get('gpus', 1)
    args.image_snapshot_ticks = config.get('snap', 50)
    args.network_snapshot_ticks = config.get('snap', 50)
    args.random_seed = config.get('seed', 0)
    metrics = config.get('metrics', ['fid50k_full'])
    if metrics is None or metrics == 'none':
        metrics = []
    args.metrics = metrics
    
    # DataLoader options
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    if config.get('workers') is not None:
        args.data_loader_kwargs.num_workers = config.get('workers')
    
    # Base network config
    cfg = config.get('cfg', 'auto')
    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2),
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8),
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
    }
    
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        spec.ref_gpus = args.num_gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(args.num_gpus * min(4096 // res, 32), 64), args.num_gpus)
        spec.mbstd = min(spec.mb // args.num_gpus, 4)
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb
        spec.ema = spec.mb * 10 / 32

    # Override with config values if provided
    if config.get('gamma') is not None:
        spec.gamma = config.get('gamma')
    if config.get('kimg') is not None:
        spec.kimg = config.get('kimg')
    if config.get('batch') is not None:
        spec.mb = config.get('batch')
        spec.mbstd = min(spec.mb // args.num_gpus, 4)
    
    # Network architecture
    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
    
    # Optimizer
    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)
    
    # Training duration and batch size
    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // args.num_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp
    
    # Special case for CIFAR
    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0
        args.loss_kwargs.style_mixing_prob = 0
        args.D_kwargs.architecture = 'orig'
    
    # Discriminator augmentation
    aug = config.get('aug', 'ada')
    if aug == 'ada':
        args.ada_target = config.get('target', 0.6)
    elif aug == 'fixed':
        if config.get('p') is None:
            raise UserError('--aug=fixed requires specifying --p')
        args.augment_p = config.get('p')
    elif aug != 'noaug':
        raise UserError(f'Unsupported augmentation mode: {aug}')
    
    # Augmentation pipeline
    augpipe = config.get('augpipe', 'bgc')
    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }
    
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])
    
    # Transfer learning
    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }
    
    resume = config.get('resume', 'noresume')
    if resume == 'noresume':
        pass
    elif resume in resume_specs:
        args.resume_pkl = resume_specs[resume]
    else:
        args.resume_pkl = resume
    
    if resume != 'noresume':
        args.ada_kimg = 100
        args.ema_rampup = None
    
    # Freezing layers for transfer learning
    freezed = config.get('freezed', 0)
    if freezed > 0:
        args.D_kwargs.block_kwargs.freeze_layers = freezed
    
    # Performance options
    if config.get('fp32', False):
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None
    
    if config.get('nhwc', False):
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True
    
    args.cudnn_benchmark = not config.get('nobench', False)
    args.allow_tf32 = config.get('allow_tf32', False)
    
    return args

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

@click.command()
@click.option('--dataset-path', type=str, help='Path to the dataset directory or zip file', required=True)
@click.option('--outdir', type=str, help='Where to save the results', required=True)
@click.option('--config', type=str, help='Path to configuration file (yaml or json)')
@click.option('--gpus', type=int, help='Number of GPUs to use', default=1)
@click.option('--snap', type=int, help='Snapshot interval in training ticks', default=50)
@click.option('--image-size', type=int, help='Override dataset\'s image resolution')
@click.option('--batch', type=int, help='Override batch size')
@click.option('--mirror', type=bool, help='Enable dataset x-flips', default=False)
@click.option('--kimg', type=int, help='Override training duration in thousands of images')
@click.option('--resume', type=str, help='Resume from given network pickle')
@click.option('--use-labels', type=bool, help='Use labels from dataset.json', default=True)
@click.option('--dry-run', is_flag=True, help='Print training options and exit')
def main(**kwargs):
    """Train a StyleGAN2-ADA model with customizable dataset.
    
    Examples:
    
    # Train with automotive dataset at 256x256 resolution
    python training_workload.py --dataset-path=./datasets/automotive --image-size=256 --outdir=./results
    
    # Train with medical dataset using a config file
    python training_workload.py --dataset-path=./datasets/medical --config=./configs/medical.yaml --outdir=./results
    
    # Resume training from a checkpoint
    python training_workload.py --dataset-path=./datasets/fashion --resume=./results/network-snapshot-000100.pkl --outdir=./results
    """
    dnnlib.util.Logger(should_flush=True)
    
    # Load config file if provided
    config = {}
    if kwargs.get('config') is not None:
        config = load_config_file(kwargs['config'])
    
    # Override config with command-line arguments
    for key, value in kwargs.items():
        if value is not None:
            if key == 'dataset_path':
                if 'dataset' not in config:
                    config['dataset'] = {}
                config['dataset']['path'] = value
            elif key == 'image_size':
                if 'dataset' not in config:
                    config['dataset'] = {}
                config['dataset']['image_size'] = value
            elif key == 'use_labels':
                if 'dataset' not in config:
                    config['dataset'] = {}
                config['dataset']['use_labels'] = value
            elif key == 'mirror':
                if 'dataset' not in config:
                    config['dataset'] = {}
                config['dataset']['mirror'] = value
            elif key != 'config':
                config[key] = value
    
    # Ensure dataset path is set
    if 'dataset' not in config or 'path' not in config['dataset']:
        raise UserError('Dataset path must be specified')
    
    # Setup training configuration
    args = setup_training_config(config)
    
    # Set output directory
    desc = f"stylegan2-ada-{Path(args.training_set_kwargs.path).stem}"
    desc += f"-{args.training_set_kwargs.resolution}x{args.training_set_kwargs.resolution}"
    args.run_dir = os.path.join(config['outdir'], desc)
    
    # Print options
    if kwargs.get('dry_run') or rank == 0:
        print('\nTraining options:')
        print(json.dumps(args, indent=2))
        print(f'Output directory: {args.run_dir}')
        print(f'Training data: {args.training_set_kwargs.path}')
        print(f'Training duration: {args.total_kimg} kimg')
        print(f'Number of GPUs: {args.num_gpus}')
        
        # Try to load the dataset to check config
        try:
            training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs)
            print(f'Dataset size: {len(training_set)} images')
            print(f'Image resolution: {training_set.resolution}x{training_set.resolution}')
            print(f'Using labels: {training_set.has_labels}')
            del training_set  # Conserve memory
        except Exception as e:
            print(f'Warning: Could not load dataset: {e}')
    
    if kwargs.get('dry_run'):
        print('Dry run; exiting.')
        return
    
    # Create output directory
    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)
    
    # Launch processes
    if args.num_gpus == 1:
        training_loop.training_loop(rank=0, **args)
    else:
        torch.multiprocessing.set_start_method('spawn')
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import torch
import random
import numpy as np
import wandb
import torch.distributed as dist
from fix_trainer_ddp import MedGMAE3DTrainer

# Only import MAE3DTrainer to avoid import errors when the other trainers aren't implemented yet

import os
import tempfile


cpu_num = 10
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)



def get_args_parser():
    parser = argparse.ArgumentParser('Gaussian MAE training script', add_help=False)
    parser.add_argument('--model_type', type=str, default="med_gmae", help='model type (med_gmae)')
    parser.add_argument('--model', type=str, default="vit_large_12p", choices=["vit_base", "vit_large", "vit_large_12p"], help='model')
    # WandB arguments
    parser.add_argument('--disable_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--proj_name', type=str, default="SelfMedMAE", help='Project name for wandb')
    parser.add_argument('--run_name', type=str, default="3MAE-96-075-128", help='Run name for wandb')
    parser.add_argument('--wandb_id', type=str, default=None, help='Run ID for resuming wandb runs')
    parser.add_argument('--dataset', type=str, default="Medical3D", help='Dataset name for wandb')
    parser.add_argument('--base_dir1', type=str, default="./dataset", help='Dataset directory')

    parser.add_argument('--data_ratio', type=float, default=1)
    parser.add_argument('--ckpt_dir', type=str, default="./ckpts/large_p12_dense_075_96_best", help='Checkpoint directory')
    parser.add_argument('--pretrained_ckpt', type=str, default="./ckpts/large_p12_dense_075_96_best/med_gmae_vit_large_12p_mix_ckpt_0499.pth", help='checkpoint path for pretraining')
    parser.add_argument('--use_pretrained', action='store_true', help='Whether to use pretrained weights')
    parser.add_argument('--base_lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument('--epochs', type=int, default=1000, help='train epoch')
    parser.add_argument('--remain', type=int, default=0, help='train epoch')
    parser.add_argument('--img_size', type=int, default=96, help='img size of per batch')
    parser.add_argument('--patch_size', type=int, default=12, help='patch size of per img')
    parser.add_argument('--in_chans', type=int, default=1, help='input channels')
    parser.add_argument('--pos_embed_type', type=str, default="sincos")
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument('--print_freq', type=int, default=50)

    parser.add_argument('--save_ckpt_dir', type=str, default=None, help='checkpoint save directory (auto-generated if not specified)')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--vis_freq', type=int, default=40, help='frequency of visualization (0 to disable)')
    parser.add_argument('--wde', type=float, default=0.2)
    parser.add_argument('--wp_ep', type=int, default=5)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--num_gaussians', type=int, default=128, help='number of Gaussians for GMAE')
    parser.add_argument('--gaussian_scale_factor', type=float, default=1.0, help='scale factor for Gaussian sizes')
    parser.add_argument('--use_image_norm', action='store_true', help='Use image-level LayerNorm for stabilizing training')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with single process')
    parser.add_argument('--max_gaussian_scale', type=float, default=0.5,
                        help='Maximum allowed scale for Gaussian parameters')
    parser.add_argument('--scale_penalty_weight', type=float, default=0,
                        help='Weight for the scale penalty term')
    parser.add_argument('--log_grads', action='store_true',
                        help='Whether to log gradient statistics')
    parser.add_argument('--log_grad_histograms', action='store_true',
                        help='Whether to log gradient histograms (every 100 steps)')
    
    parser.add_argument('--use_residual_gaussians', action='store_true',
                        help='Whether to use residual gaussians')
    parser.add_argument('--num_residuals', type=int, default=13,
                        help='Number of residual MLPs')
    parser.add_argument('--residual_hidden_dim', type=int, default=256,
                        help='Hidden dimension for residual MLPs')

    parser.add_argument('--train_residual_only', action='store_true',
                        help='Whether to train only residual part first')
    parser.add_argument('--residual_epochs', type=int, default=100,
                        help='Number of epochs for residual-only training')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='Checkpoint save frequency (epochs)')

    parser.add_argument('--use_hierarchical_gaussians', action='store_true',
                        help='Whether to use hierarchical gaussian decoder')
    parser.add_argument('--hierarchical_level1_ratio', type=int, default=12,
                        help='Level 1 expansion ratio')
    parser.add_argument('--hierarchical_level2_ratio', type=int, default=7,
                        help='Level 2 expansion ratio')

    parser.add_argument('--decoder_type', type=str, default='standard',
                        choices=['standard', 'residual', 'hierarchical', 'cgs'],
                        help='Decoder type: standard, residual, hierarchical, cgs')

    parser.add_argument('--cgs_layers', nargs='+', type=int, default=[2, 5, 8, 11],
                        help='CGS layer indices')
    parser.add_argument('--cgs_upsample_ratios', nargs='+', type=int, default=[1, 2, 4, 8],
                        help='CGS upsample ratios')

    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Whether to freeze encoder parameters')

    return parser

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    args, unknown = get_args_parser().parse_known_args()

    if args.save_ckpt_dir is None:
        base_ckpt_path = "./ckpts"
        mask_ratio_str = f"{int(args.mask_ratio * 100):02d}"
        dir_name = f"{args.model}_{args.img_size}_mask{mask_ratio_str}_g{args.num_gaussians}"

        if args.use_residual_gaussians:
            dir_name += f"_res{args.num_residuals}"

        if args.use_hierarchical_gaussians:
            dir_name += f"_hier_l1_{args.hierarchical_level1_ratio}_l2_{args.hierarchical_level2_ratio}"

        if args.decoder_type != 'standard':
            dir_name += f"_{args.decoder_type}"

        args.save_ckpt_dir = os.path.join(base_ckpt_path, dir_name)
        print(f"Auto-generated save directory: {args.save_ckpt_dir}")

    # Check if ckpt_dir exists and create it if not
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        print(f"Created checkpoint directory: {args.ckpt_dir}")
        
    # Check if save_ckpt_dir exists and create it if not
    if not os.path.exists(args.save_ckpt_dir):
        os.makedirs(args.save_ckpt_dir, exist_ok=True)
        print(f"Created save checkpoint directory: {args.save_ckpt_dir}")
    
    # If pretrained_ckpt is empty and use_pretrained is not specified, turn off pretraining
    if not args.pretrained_ckpt and not args.use_pretrained:
        print("No pretrained checkpoint specified, training from scratch")
    
    # Verify that pretrained checkpoint exists if specified
    if args.pretrained_ckpt:
        ckpt_path = os.path.join(args.ckpt_dir, args.pretrained_ckpt)
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Pretrained checkpoint not found at {ckpt_path}")
            print("Will train from scratch instead")
            args.pretrained_ckpt = ""
    
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    seed_torch(args.seed)
    
    if args.model_type == "med_gmae":
        trainer = MedGMAE3DTrainer(args=args)
        print(f"Using MedGMAE3DTrainer for model type: {args.model_type}")

    print("\nNOTICE: Trainer setup complete, starting training...\n")
    print("="*80 + "\n")

    try:
        trainer.run()
    finally:
        if not args.disable_wandb and dist.get_rank() == 0:
            try:
                if wandb.run is not None:
                    print("Closing wandb...")
                    wandb.finish()
            except Exception as e:
                print(f"Error closing wandb: {e}")

import os
import time
import math
import shutil
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from dataloader.med_datasets import LargeMedicalDataSets
from models.fix.mae3d import build_vit_base_mae_3d, build_vit_large_mae_3d, build_vit_large_mae_p12_3d
from tqdm import tqdm
import torch.distributed as dist
sys.path.append('./')
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.visualization import patches3d_to_grid
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import wandb

class MAE3DTrainer(nn.Module):
    """
    3D Masked Autoencoder Trainer
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_name = 'MAE3D'
        dist.init_process_group('nccl', init_method='env://')
        rank = dist.get_rank()   
        local_rank = int(os.environ['LOCAL_RANK'])
        master_addr = os.environ['MASTER_ADDR']  
        master_port = os.environ['MASTER_PORT']  
        print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
        
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            print(f"Warning: local_rank={local_rank} exceeds available GPU count {device_count}, falling back to GPU 0")
            local_rank = 0
            
        torch.cuda.set_device(local_rank)
        args.local_rank = local_rank
        
        self.use_wandb = not getattr(args, 'disable_wandb', True) and dist.get_rank() == 0
        
        if self.use_wandb:
            if args.wandb_id is None:
                args.wandb_id = wandb.util.generate_id()
            if args.run_name is None:
                args.run_name = f'{self.model_name}_{args.model}'
                
            wandb.init(
                project=f"{args.proj_name}_{args.dataset}",
                name=args.run_name,
                config=vars(args),
                id=args.wandb_id,
                resume='allow'
            )
            print(f"Initialized wandb with run ID: {args.wandb_id}")
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.init_lr()
        self.build_model()
        self.build_optimizer()
        self.build_dataloader()
        self.iters_per_epoch = len(self.dataloader)
        if not os.path.exists(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
        print("mask ratio: {}".format(args.mask_ratio))
        if dist.get_rank() == 0:
            print(f"Local Batch Size on GPU {args.local_rank}: {args.batch_size}")
    def init_lr(self):
        args = self.args
        # infer learning rate before changing batch size
        self.lr = args.base_lr

    def build_model(self):
        args = self.args
        if args.model == "vit_base":
            model = build_vit_base_mae_3d(args=args).to(args.local_rank)
        elif args.model == "vit_large":
            model = build_vit_large_mae_3d(args=args).to(args.local_rank)
        elif args.model == "vit_large_12p":
            model = build_vit_large_mae_p12_3d(args=args).to(args.local_rank)
        else:
            model = None
        if os.path.exists(os.path.join(args.ckpt_dir, args.pretrained_ckpt)) and dist.get_rank() == 0:
            model_dict = torch.load(os.path.join(args.ckpt_dir, args.pretrained_ckpt))["state_dict"]
            pretrained_dict = {}
            for k, v in model_dict.items():
                print("k: ", k)
                if k.startswith("module."):
                    pretrained_dict[k[7:]] = v
                else:
                    pretrained_dict[k] = v
            missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
            print(f"Model weights loaded, missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
            if model.load_state_dict(pretrained_dict, strict=False):
                print("Load ckpt {} Successful !".format(os.path.join(args.ckpt_dir, args.pretrained_ckpt)))

        self.model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        
    def build_optimizer(self):
        args = self.args
        #optim_params = self.get_parameter_groups()
        
        optim_params = self.get_param_groups(nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'decoder_pos_embed', 'gamma'})
        
        # TODO: create optimizer factory
        self.optimizer = torch.optim.AdamW(optim_params,
                                            lr=self.lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)
    def build_dataloader(self):
        args = self.args
        train_dataset = LargeMedicalDataSets(args=args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                    batch_size=args.batch_size, num_workers=16, prefetch_factor=2, pin_memory=True)
        print("num_workers: {}, prefetch_factor: {}, bs: {}".format(16, 2, args.batch_size * 8 * 5))
        
        if getattr(args, 'vis_freq', 0) > 0:
            val_dataset = LargeMedicalDataSets(args=args, is_val=True)
            self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                           num_workers=4, pin_memory=True)

    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = 0
        try:
            for epoch in range(args.remain, args.epochs):
                # train for one epoch
                self.dataloader.sampler.set_epoch(epoch)
                niters = self.epoch_train(epoch, niters)
                
                if getattr(args, 'vis_freq', 0) > 0 and hasattr(self, 'val_dataloader') and \
                   (epoch == 0 or (epoch + 1) % args.vis_freq == 0):
                    print(f"=> start visualizing after {epoch + 1} epochs")
                    self.vis_reconstruction(niters)
                    print("=> finish visualizing")
                
                if (epoch + 1) % 20 == 0 and dist.get_rank() == 0:
                    checkpoint_path = f'{args.save_ckpt_dir}/{args.model}_mix_ckpt_{epoch:04d}.pth'
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scaler': self.scaler.state_dict(),  # additional line compared with base imple
                    }, is_best=False, filename=checkpoint_path)
                    print("=> finish saving checkpoint")
                    
                    # if self.use_wandb:
                    #     wandb.save(checkpoint_path)
        except Exception as e:
            print(f"Training error: {e}")
            raise e

           

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        total_loss = AverageMeter()
        load_time = 0
        forward_time = 0
        bp_time = 0
        cache2gpu_time = 0
        # switch to train mode
        model.train()
        load_start_time = time.time()
        start_time = time.time()
        if dist.get_rank() == 0:
            #pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', unit='batch')
            #for i, batch_data in enumerate(pbar):
            for i, batch_data in enumerate(train_loader):
                load_time += time.time() - load_start_time
                # adjust learning at the beginning of each iteration
                self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)
                
                # For SSL pretraining, only image data is required for training
                gpu_time = time.time()
                #image = torch.stack([crop_per_batch["image"] for crop_per_batch in batch_data], dim=0).view(-1, 1, 48, 48, 48)
                
                
                batch_data = [{k: v.to(args.local_rank) for k, v in crop.items()} for crop in batch_data]
                image = torch.stack([crop["image"] for crop in batch_data], dim=0).view(-1, 1, 96, 96, 96).to(args.local_rank)
                # gpu_time = time.time()
                #print(image.shape, len(batch_data))
                #image = image.to(args.local_rank)
                cache2gpu_time += time.time() - gpu_time
                #print(cache2gpu_time)
                
                
                # compute output and loss
                forward_start_time = time.time()
                with torch.cuda.amp.autocast(True):
                    loss = model(image, return_image=False)
                    total_loss.update(loss.item(), image.size(0))
                forward_time += time.time() - forward_start_time

                # compute gradient and do SGD step
                bp_start_time = time.time()
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                bp_time += time.time() - bp_start_time
                niters += 1
                load_start_time = time.time()
                
                if self.use_wandb and i % 10 == 0:
                    wandb.log({
                        'iter/loss': loss.item(),
                        'iter/learning_rate': optimizer.param_groups[0]['lr'],
                        'iter/time_data_loading': load_time / (i+1),
                        'iter/time_gpu_transfer': cache2gpu_time / (i+1),
                        'iter/time_forward': forward_time / (i+1),
                        'iter/time_backward': bp_time / (i+1),
                        'epoch': epoch,
                        'iter': niters
                    }, step=model.module.global_step)
                
                #pbar.set_postfix({'Loss': total_loss.avg})
        else:
            for i, batch_data in enumerate(train_loader): #enumerate(pbar):
                load_time += time.time() - load_start_time
                # adjust learning at the beginning of each iteration
                self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)
                
                # For SSL pretraining, only image data is required for training
                gpu_time = time.time()
                batch_data = [{k: v.to(args.local_rank) for k, v in crop.items()} for crop in batch_data]
                image = torch.stack([crop["image"] for crop in batch_data], dim=0).view(-1, 1, 96, 96, 96).to(args.local_rank)
                cache2gpu_time += time.time() - gpu_time
                #print(cache2gpu_time)
                
                
                # compute output and loss
                forward_start_time = time.time()
                with torch.cuda.amp.autocast(True):
                    loss = model(image, return_image=False)
                    total_loss.update(loss.item(), image.size(0))
                forward_time += time.time() - forward_start_time

                # compute gradient and do SGD step
                bp_start_time = time.time()
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                bp_time += time.time() - bp_start_time
                niters += 1
                load_start_time = time.time()
                
                #pbar.set_postfix({'Loss': total_loss.avg})
            
        end_time = time.time()
        duration = end_time - start_time
        # Log to the screen
        if dist.get_rank() == 0:
            print("Epoch: {}/{} took {:.2f} seconds | TotalIter {}/{} | Init Lr: {:.6f} | Lr: {:.6f} | Load Time: {:.2f} s | GPU Time: {:.2f} s | Forward Time: {:.2f} s | Backward Time: {:.2f} s | Loss: {:.4f}"
                  .format(epoch, args.epochs, duration, niters, args.epochs * self.iters_per_epoch, self.lr, optimizer.param_groups[0]['lr'], 
                          load_time, 
                          cache2gpu_time,
                          forward_time, 
                          bp_time, 
                          total_loss.avg))
            
            if self.use_wandb:
                wandb.log({
                    'epoch/loss': total_loss.avg,
                    'epoch/learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch/time_total': duration,
                    'epoch/time_data_loading': load_time,
                    'epoch/time_gpu_transfer': cache2gpu_time,
                    'epoch/time_forward': forward_time,
                    'epoch/time_backward': bp_time,
                    'epoch': epoch
                }, step=model.module.global_step)
        return niters



    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        torch.save(state, filename)
        if is_best:
            best_path = os.path.join(os.path.dirname(filename), 'model_best.pth')
            shutil.copyfile(filename, best_path)
            # Log best model to wandb
            if self.use_wandb and dist.get_rank() == 0:
                wandb.save(best_path)

    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])  # additional line compared with base imple
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            # cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
            decay_factor = 0.9 ** (epoch - args.warmup_epochs)
            cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))) * decay_factor

        for param_group in self.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr

    def lr_wd_annealing(self, peak_lr, wd, wd_end, cur_it, wp_it, max_it):
        wp_it = round(wp_it)
        if cur_it < wp_it:
            cur_lr = 0.005 * peak_lr + 0.995 * peak_lr * cur_it / wp_it
        else:
            ratio = (cur_it - wp_it) / (max_it - 1 - wp_it)
            cur_lr = 0.001 * peak_lr + 0.999 * peak_lr * (0.5 + 0.5 * math.cos(math.pi * ratio))
        
        ratio = cur_it / (max_it - 1)
        cur_wd = wd_end + (wd - wd_end) * (0.5 + 0.5 * math.cos(math.pi * ratio))
        
        min_lr, max_lr = cur_lr, cur_lr
        min_wd, max_wd = cur_wd, cur_wd
        for param_group in self.optimizer.param_groups:
            scaled_lr = param_group['lr'] = cur_lr * param_group.get('lr_scale', 1)  # 'lr_scale' could be assigned
            min_lr, max_lr = min(min_lr, scaled_lr), max(max_lr, scaled_lr)
            scaled_wd = param_group['weight_decay'] = cur_wd * param_group.get('weight_decay_scale', 1)  # 'weight_decay_scale' could be assigned
            min_wd, max_wd = min(min_wd, scaled_wd), max(max_wd, scaled_wd)
        # return min_lr, max_lr, min_wd, max_wd

    def get_param_groups(self, nowd_keys=()):
        para_groups, para_groups_dbg = {}, {}
    
        for name, para in self.model.named_parameters():
            if not para.requires_grad:
                continue  # frozen weights
            if len(para.shape) == 1 or name.endswith('.bias') or any(k in name for k in nowd_keys):
                wd_scale, group_name = 0., 'no_decay'
            else:
                wd_scale, group_name = 1., 'decay'
            
            if group_name not in para_groups:
                para_groups[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
                para_groups_dbg[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
            para_groups[group_name]['params'].append(para)
            para_groups_dbg[group_name]['params'].append(name)
        
        return list(para_groups.values())
    
    

    def group_params(self, model, fix_lr=False):
        all_params = set(model.parameters())
        wd_params = set()
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                wd_params.add(m.weight)
        no_wd = all_params - wd_params
        params_group = [{'params': list(wd_params), 'fix_lr': fix_lr},
                        {'params': list(no_wd), 'weight_decay': 0., 'fix_lr': fix_lr}]
        return params_group

    def get_parameter_groups(self, get_layer_id=None, get_layer_scale=None, verbose=False):
        args = self.args
        weight_decay = args.weight_decay
        model = self.model

        if hasattr(model, 'no_weight_decay'):
            skip_list = model.no_weight_decay()
        else:
            skip_list = {}

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if get_layer_id is not None:
                layer_id = get_layer_id(name)
                group_name = "layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if get_layer_scale is not None:
                    scale = get_layer_scale(layer_id)
                else:
                    scale = 1.

                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        if verbose:
            import json
            print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        else:
            print("Param groups information is omitted...")
        return list(parameter_group_vars.values())




from models.fix.med_gmae3d import build_vit_base_med_gmae_3d, build_vit_large_med_gmae_3d, \
    build_vit_large_med_gmae_p12_3d




# Add this class to the end of the file
class MedGMAE3DTrainer(nn.Module):
    """
    3D Medical Gaussian Masked Autoencoder Trainer
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_name = 'MedGMAE3D'
        dist.init_process_group('nccl', init_method='env://')
        rank = dist.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
        
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            print(f"Warning: local_rank={local_rank} exceeds available GPU count {device_count}, falling back to GPU 0")
            local_rank = 0
            
        torch.cuda.set_device(local_rank)
        args.local_rank = local_rank
        
        self.use_wandb = not getattr(args, 'disable_wandb', True) and dist.get_rank() == 0
        
        if self.use_wandb:
            if args.wandb_id is None:
                args.wandb_id = wandb.util.generate_id()
            if args.run_name is None:
                args.run_name = f'{self.model_name}_{args.model}'
                
            wandb.init(
                project=f"{args.proj_name}_{args.dataset}",
                name=args.run_name,
                config=vars(args),
                id=args.wandb_id,
                resume='allow'
            )
            print(f"Initialized wandb with run ID: {args.wandb_id}")

        self.scaler = torch.cuda.amp.GradScaler()
        self.init_lr()
        self.build_model()
        self.build_optimizer()
        self.build_dataloader()
        self.iters_per_epoch = len(self.dataloader)
        if not os.path.exists(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
        print("mask ratio: {}".format(args.mask_ratio))
        if dist.get_rank() == 0:
            print(f"Local Batch Size on GPU {args.local_rank}: {args.batch_size}")

    def init_lr(self):
        args = self.args
        # infer learning rate before changing batch size
        self.lr = args.base_lr

    def build_model(self):
        args = self.args
        
        if args.model == "vit_base":
            model = build_vit_base_med_gmae_3d(args=args).to(args.local_rank)
        elif args.model == "vit_large":
            model = build_vit_large_med_gmae_3d(args=args).to(args.local_rank)
        elif args.model == "vit_large_12p":
            model = build_vit_large_med_gmae_p12_3d(args=args).to(args.local_rank)
        else:
            model = None
            
        model.use_wandb = self.use_wandb
        model._trainer_wandb_log = self.log_model_stats

        if args.pretrained_ckpt and os.path.exists(os.path.join(args.ckpt_dir, args.pretrained_ckpt)) and dist.get_rank() == 0:
            model_dict = torch.load(os.path.join(args.ckpt_dir, args.pretrained_ckpt))["state_dict"]
            pretrained_dict = {}
            
            base_model_prefixes = [
                'encoder.',
                'encoder_to_decoder.',
                'gaussian_query_tokens',
                'decoder.base_decoder.',
                'decoder.query_embed',
                'decoder.num_gaussians',
            ]
            
            exclude_prefixes = [
                'decoder.residual_position_heads',
                'decoder.residual_scale_heads', 
                'decoder.residual_rotation_heads',
                'decoder.residual_density_heads',
                'decoder.level1_position_heads',
                'decoder.level1_scale_heads',
                'decoder.level1_rotation_heads', 
                'decoder.level1_density_heads',
                'decoder.level2_position_heads',
                'decoder.level2_scale_heads',
                'decoder.level2_rotation_heads',
                'decoder.level2_density_heads',
            ]
            
            loaded_keys = []
            skipped_keys = []
            
            for k, v in model_dict.items():
                clean_key = k[7:] if k.startswith("module.") else k
                
                should_exclude = any(clean_key.startswith(prefix) for prefix in exclude_prefixes)
                
                if should_exclude:
                    skipped_keys.append(clean_key)
                    continue
                
                is_base_weight = any(clean_key.startswith(prefix) for prefix in base_model_prefixes)
                
                if not is_base_weight:
                    has_residual_keywords = any(keyword in clean_key.lower() for keyword in 
                                              ['residual', 'level1', 'level2', 'hierarchical'])
                    if not has_residual_keywords:
                        is_base_weight = True
                
                if is_base_weight:
                    pretrained_dict[clean_key] = v
                    loaded_keys.append(clean_key)
                else:
                    skipped_keys.append(clean_key)
                    
            missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)

            print(f"\n{'='*60}")
            print(f"Pretrained weights loaded:")
            print(f"  Loaded weights count: {len(loaded_keys)}")
            print(f"  Skipped weights count: {len(skipped_keys)}")
            print(f"  Missing weights count: {len(missing_keys)}")
            print(f"  Unexpected weights count: {len(unexpected_keys)}")

            if len(loaded_keys) > 0:
                print(f"  Successfully loaded base model weights")
                print(f"  Example loaded weights: {loaded_keys[:3]}...")

            if len(skipped_keys) > 0:
                print(f"  Example skipped weights (residual/hierarchical): {skipped_keys[:3]}...")

            print(f"{'='*60}\n")
                
        if getattr(args, 'use_hierarchical_gaussians', False):
            from models.fix.networks.med_gaussian_decoder import enhance_gsmae_with_hierarchical_residuals

            original_num_gaussians = model.num_gaussians
            print(f"\n{'='*40}")
            print(f"Enhancing model with hierarchical residual Gaussian decoder")
            print(f"Level1 expansion ratio: {getattr(args, 'hierarchical_level1_ratio', 3)}")
            print(f"Level2 expansion ratio: {getattr(args, 'hierarchical_level2_ratio', 16)}")

            model = enhance_gsmae_with_hierarchical_residuals(
                model=model,
                level1_ratio=getattr(args, 'hierarchical_level1_ratio', 3),
                level2_ratio=getattr(args, 'hierarchical_level2_ratio', 16)
            )

            print(f"Gaussian count increased from {original_num_gaussians} to {model.num_gaussians}")
            print(f"{'='*40}\n")
            
        elif hasattr(args, 'use_residual_gaussians') and args.use_residual_gaussians:
            from models.fix.networks.med_gaussian_decoder import enhance_gsmae_with_residuals

            original_num_gaussians = model.num_gaussians
            print(f"\n{'='*40}")
            print(f"Enhancing model with residual Gaussian decoder")
            print(f"Number of residual MLPs: {args.num_residuals}")
            print(f"Hidden layer dimension: {args.residual_hidden_dim}")

            model = enhance_gsmae_with_residuals(
                model=model,
                num_residuals=args.num_residuals,
                hidden_dim=args.residual_hidden_dim
            )

            print(f"Gaussian count increased from {original_num_gaussians} to {model.num_gaussians}")
            print(f"{'='*40}\n")

        self.model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    def build_optimizer(self):
        args = self.args
        optim_params = self.get_param_groups(
            nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'decoder_pos_embed', 'gamma', 'gaussian_query_tokens'})

        self.optimizer = torch.optim.AdamW(optim_params,
                                           lr=self.lr,
                                           betas=(args.beta1, args.beta2),
                                           weight_decay=args.weight_decay)

    def build_dataloader(self):
        args = self.args
        train_dataset = LargeMedicalDataSets(args=args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                     batch_size=args.batch_size, num_workers=16, prefetch_factor=2, pin_memory=True)
        print("num_workers: {}, prefetch_factor: {}, bs: {}".format(16, 2, args.batch_size * 8 * 5))
        
        if getattr(args, 'vis_freq', 0) > 0:
            val_dataset = LargeMedicalDataSets(args=args, is_val=True)
            self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                           num_workers=4, pin_memory=True)

    def freeze_encoder_params(self):
        """Freeze encoder parameters, keep only decoder trainable"""
        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            if name.startswith('module.encoder.') or 'encoder' in name:
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
                trainable_count += 1

        if dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f"Encoder parameters frozen: frozen params: {frozen_count}, trainable params: {trainable_count}")
            print(f"Only decoder parameters can be optimized")
            print(f"{'='*60}\n")

        return frozen_count, trainable_count

    def freeze_base_model_params(self):
        """Freeze base model parameters, keep only residual Gaussian parts trainable"""
        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            if 'residual_position_heads' not in name and 'residual_scale_heads' not in name and \
               'residual_rotation_heads' not in name and 'residual_density_heads' not in name:
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
                trainable_count += 1

        if dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f"Model parameters frozen: frozen params: {frozen_count}, trainable params: {trainable_count}")
            print(f"Only residual Gaussian parameters can be optimized")
            print(f"{'='*60}\n")

        return frozen_count, trainable_count

    def unfreeze_all_params(self):
        """Unfreeze all model parameters"""
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            unfrozen_count += 1

        if dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f"Model parameters unfrozen: unfrozen {unfrozen_count} parameters")
            print(f"All parameters are now trainable")
            print(f"{'='*60}\n")

        return unfrozen_count
            
    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = 0
        
        if getattr(args, 'freeze_encoder', False):
            if dist.get_rank() == 0:
                print(f"\n{'='*80}")
                print(f"Freeze encoder mode: train only decoder part")
                print(f"{'='*80}\n")
            
            self.freeze_encoder_params()
            
            self.build_optimizer()
        
        use_phased_training = hasattr(args, 'train_residual_only') and args.train_residual_only and \
                              hasattr(args, 'residual_epochs') and args.residual_epochs > 0 and \
                              (hasattr(args, 'use_residual_gaussians') and args.use_residual_gaussians or \
                               hasattr(args, 'use_hierarchical_gaussians') and args.use_hierarchical_gaussians)
        
        try:
            if use_phased_training:
                if dist.get_rank() == 0:
                    print(f"\n{'='*80}")
                    print(f"Phase 1: Freeze base model, train only residual parts ({args.residual_epochs} epochs)")
                    print(f"{'='*80}\n")
                
                self.freeze_base_model_params()
                
                self.build_optimizer()
                
                for epoch in range(args.remain, args.residual_epochs):
                    self.dataloader.sampler.set_epoch(epoch)
                    
                    niters = self.epoch_train(epoch, niters, phase='residual')
                    
                    if getattr(args, 'vis_freq', 0) > 0 and hasattr(self, 'val_dataloader') and \
                       (epoch == 0 or (epoch + 1) % args.vis_freq == 0):
                        if dist.get_rank() == 0:
                            print(f"=> Phase 1: Starting visualization after {epoch + 1} epochs")
                        self.vis_reconstruction(niters)
                        if dist.get_rank() == 0:
                            print("=> Visualization complete")
                    
                    if (epoch + 1) % getattr(args, 'save_freq', 20) == 0 and dist.get_rank() == 0:
                        checkpoint_path = f'{args.save_ckpt_dir}/med_gmae_{args.model}_residual_phase1_ckpt_{epoch:04d}.pth'
                        self.save_checkpoint({
                            'epoch': epoch + 1,
                            'phase': 'residual',
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(),
                        }, is_best=False, filename=checkpoint_path)
                        print("=> Phase 1 checkpoint saved")

                if dist.get_rank() == 0:
                    print(f"\n{'='*80}")
                    print(f"Phase 2: Unfreeze all parameters for finetuning ({args.epochs - args.residual_epochs} epochs)")
                    print(f"{'='*80}\n")
                
                self.unfreeze_all_params()
                
                self.build_optimizer()
                
                for epoch in range(args.residual_epochs, args.epochs):
                    self.dataloader.sampler.set_epoch(epoch)
                    
                    niters = self.epoch_train(epoch, niters, phase='finetune')
                    
                    if getattr(args, 'vis_freq', 0) > 0 and hasattr(self, 'val_dataloader') and \
                       ((epoch - args.residual_epochs) == 0 or (epoch + 1) % args.vis_freq == 0):
                        if dist.get_rank() == 0:
                            print(f"=> Phase 2: Starting visualization after {epoch + 1} epochs")
                        self.vis_reconstruction(niters)
                        if dist.get_rank() == 0:
                            print("=> Visualization complete")
                    
                    if (epoch + 1) % getattr(args, 'save_freq', 20) == 0 and dist.get_rank() == 0:
                        checkpoint_path = f'{args.save_ckpt_dir}/med_gmae_{args.model}_finetune_ckpt_{epoch:04d}.pth'
                        self.save_checkpoint({
                            'epoch': epoch + 1,
                            'phase': 'finetune',
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(),
                        }, is_best=False, filename=checkpoint_path)
                        print("=> Phase 2 checkpoint saved")
            
            else:
                for epoch in range(args.remain, args.epochs):
                    # train for one epoch
                    self.dataloader.sampler.set_epoch(epoch)
                    niters = self.epoch_train(epoch, niters)
                    
                    if getattr(args, 'vis_freq', 0) > 0 and hasattr(self, 'val_dataloader') and \
                       (epoch == 0 or (epoch + 1) % args.vis_freq == 0):
                        print(f"=> start visualizing after {epoch + 1} epochs")
                        self.vis_reconstruction(niters)
                        print("=> finish visualizing")
                    
                    if (epoch + 1) % getattr(args, 'save_freq', 20) == 0 and dist.get_rank() == 0:
                        checkpoint_path = f'{args.save_ckpt_dir}/med_gmae_{args.model}_mix_ckpt_{epoch:04d}.pth'
                        self.save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(),
                        }, is_best=False, filename=checkpoint_path)
                        print("=> finish saving checkpoint")
        except Exception as e:
            print(f"Training error: {e}")
            raise e

    def epoch_train(self, epoch, niters, phase=None):
        args = self.args
        train_loader = self.dataloader
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        total_loss = AverageMeter()
        load_time = 0
        forward_time = 0
        bp_time = 0
        cache2gpu_time = 0
        
        phase_name = f"[{phase}]" if phase else ""
        
        # switch to train mode
        model.train()
        load_start_time = time.time()
        start_time = time.time()

        for i, batch_data in enumerate(train_loader):
            load_time += time.time() - load_start_time
            # adjust learning at the beginning of each iteration
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            # Process input data
            gpu_time = time.time()
            batch_data = [{k: v.to(args.local_rank) for k, v in crop.items()} for crop in batch_data]
            image = torch.stack([crop["image"] for crop in batch_data], dim=0).view(-1, 1, 96, 96, 96).to(
                args.local_rank)
            cache2gpu_time += time.time() - gpu_time


            if epoch < 0 and i % 100 == 0:
                from models.fix.med_gmae3d import patchify_image, batched_shuffle_indices
                import numpy as np
                in_chans = 1
                with torch.no_grad():
                    _, _, rendered_image, _, _ = model(image, return_image=True)
                    
                    residual_decoder = model.module.decoder
                    base_decoder = residual_decoder.base_decoder
                    
                    batch_size = image.size(0)
                    x_patches = patchify_image(image, model.module.patch_size)
                    
                    length = np.prod(model.module.grid_size)
                    mask_ratio = model.module.args.mask_ratio
                    sel_length = int(length * (1 - mask_ratio))
                    
                    shuffle_indices = batched_shuffle_indices(batch_size, length, device=image.device)
                    sel_indices = shuffle_indices[:, :sel_length]
                    
                    sel_encoder_pos_embed = model.module.encoder_pos_embed.expand(batch_size, -1, -1)\
                        .gather(dim=1, index=sel_indices[:, :, None].expand(-1, -1, model.module.encoder.embed_dim))
                    
                    shuffled_x = x_patches.gather(dim=1, index=shuffle_indices[:, :, None].expand(
                        -1, -1, in_chans * np.prod(model.module.patch_size)))
                    sel_x = shuffled_x[:, :sel_length, :]
                    
                    encoder_outputs = model.module.encoder(sel_x, sel_encoder_pos_embed)
                    encoder_outputs = model.module.encoder_to_decoder(encoder_outputs)
                    
                    query_tokens = model.module.gaussian_query_tokens.expand(batch_size, -1, -1)
                    all_tokens = torch.cat([encoder_outputs[:, :1], query_tokens, encoder_outputs[:, 1:]], dim=1)
                    
                    base_positions, base_scales, base_rotations, base_densities = base_decoder(all_tokens)
                    
                    gaussian_tokens = all_tokens[:, 1:residual_decoder.num_gaussians+1]
                    
                    if hasattr(residual_decoder, 'residual_position_heads'):
                        residual_pos = residual_decoder.residual_position_heads[0](gaussian_tokens)
                        residual_scale = residual_decoder.residual_scale_heads[0](gaussian_tokens)
                        residual_rot = residual_decoder.residual_rotation_heads[0](gaussian_tokens)
                        residual_density = residual_decoder.residual_density_heads[0](gaussian_tokens)
                    
                        print(f"\n============= Batch {i} Parameter Stats =============")
                        print(f"Base position mean: {base_positions.mean().item():.6f}, range: [{base_positions.min().item():.6f}, {base_positions.max().item():.6f}]")
                        print(f"Base scale mean: {base_scales.mean().item():.6f}, range: [{base_scales.min().item():.6f}, {base_scales.max().item():.6f}]")

                        print(f"Position residual mean: {residual_pos.mean().item():.6f}, range: [{residual_pos.min().item():.6f}, {residual_pos.max().item():.6f}]")
                        print(f"Scale residual mean: {residual_scale.mean().item():.6f}, range: [{residual_scale.min().item():.6f}, {residual_scale.max().item():.6f}]")
                        print(f"Rotation residual mean: {residual_rot.mean().item():.6f}, range: [{residual_rot.min().item():.6f}, {residual_rot.max().item():.6f}]")
                        print(f"Density residual mean: {residual_density.mean().item():.6f}, range: [{residual_density.min().item():.6f}, {residual_density.max().item():.6f}]")

                        print(f"Position residual relative ratio: {(residual_pos.abs().mean() / base_positions.abs().mean()).item():.6f}")
                        print(f"Scale residual relative ratio: {(residual_scale.abs().mean() / base_scales.abs().mean()).item():.6f}")

                        positions, scales, rotations, densities = residual_decoder(all_tokens)
                        print(f"Mixed position mean: {positions.mean().item():.6f}, range: [{positions.min().item():.6f}, {positions.max().item():.6f}]")
                        print(f"Mixed scale mean: {scales.mean().item():.6f}, range: [{scales.min().item():.6f}, {scales.max().item():.6f}]")

                        print(f"Base Gaussian count: {base_positions.shape[1]}, mixed Gaussian count: {positions.shape[1]}")
                        print(f"Theoretical expansion ratio: {model.module.decoder.num_residuals + 1}")

                        with torch.no_grad():
                            w_norm = residual_decoder.residual_position_heads[0].weight.norm().item()
                            b_norm = residual_decoder.residual_position_heads[0].bias.norm().item()
                            print(f"Residual MLP weight norm: {w_norm:.6f}, bias norm: {b_norm:.6f}")

                        print(f"==========================================\n")

            # compute output and loss
            forward_start_time = time.time()
            with torch.cuda.amp.autocast(True):
                loss = model(image, return_image=False)
                total_loss.update(loss.item(), image.size(0))
                
                if hasattr(model.module, 'current_loss_stats'):
                    loss_stats = model.module.current_loss_stats
                    param_stats = model.module.current_param_stats
                    geom_stats = model.module.current_geom_stats
                    
                    grad_stats = {}
                    if hasattr(model.module, 'current_grad_stats'):
                        grad_stats = model.module.current_grad_stats
                    
                    if self.use_wandb and dist.get_rank() == 0 and i % 10 == 0:
                        try:
                            import wandb
                            if wandb.run is not None:
                                log_dict = {}
                                
                                for key, val in loss_stats.items():
                                    if key != 'global_step':
                                        log_dict[f'loss/{key}'] = val
                                
                                for key, val in param_stats.items():
                                    log_dict[f'params/{key}'] = val
                                
                                for key, val in geom_stats.items():
                                    log_dict[f'geometry/{key}'] = val
                                
                                for key, val in grad_stats.items():
                                    log_dict[f'grad/{key}'] = val
                                
                                wandb.log(log_dict, step=model.module.global_step)
                                
                                if hasattr(model.module, 'current_loss_stats'):
                                    del model.module.current_loss_stats
                                if hasattr(model.module, 'current_param_stats'):
                                    del model.module.current_param_stats
                                if hasattr(model.module, 'current_geom_stats'):
                                    del model.module.current_geom_stats
                                if hasattr(model.module, 'current_grad_stats'):
                                    del model.module.current_grad_stats
                        except Exception as e:
                            print(f"Failed to log batch statistics: {e}")
            forward_time += time.time() - forward_start_time

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time += time.time() - bp_start_time
            niters += 1
            load_start_time = time.time()
            
            if self.use_wandb and i % 10 == 0:
                wandb.log({
                    'iter/loss': loss.item(),
                    'iter/learning_rate': optimizer.param_groups[0]['lr'],
                    'iter/time_data_loading': load_time / (i+1) if i > 0 else 0,
                    'iter/time_gpu_transfer': cache2gpu_time / (i+1) if i > 0 else 0,
                    'iter/time_forward': forward_time / (i+1) if i > 0 else 0,
                    'iter/time_backward': bp_time / (i+1) if i > 0 else 0,
                    'epoch': epoch,
                    'iter': niters
                }, step=model.module.global_step)
        
        end_time = time.time()
        duration = end_time - start_time
        # Log to the screen
        if dist.get_rank() == 0:
            print(
                "Epoch: {}/{} {} took {:.2f} seconds | TotalIter {}/{} | Init Lr: {:.6f} | Lr: {:.6f} | Load Time: {:.2f} s | GPU Time: {:.2f} s | Forward Time: {:.2f} s | Backward Time: {:.2f} s | Loss: {:.12f}"
                .format(epoch, args.epochs, phase_name, duration, niters, args.epochs * self.iters_per_epoch, self.lr,
                        optimizer.param_groups[0]['lr'],
                        load_time,
                        cache2gpu_time,
                        forward_time,
                        bp_time,
                        total_loss.avg))
            
            if self.use_wandb:
                log_dict = {
                    'epoch/loss': total_loss.avg,
                    'epoch/learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch/time_total': duration,
                    'epoch/time_data_loading': load_time,
                    'epoch/time_gpu_transfer': cache2gpu_time,
                    'epoch/time_forward': forward_time,
                    'epoch/time_backward': bp_time,
                    'epoch': epoch
                }
                
                if phase:
                    log_dict['epoch/phase'] = phase
                
                wandb.log(log_dict, step=model.module.global_step)
        return niters

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        torch.save(state, filename)
        if is_best:
            best_path = os.path.join(os.path.dirname(filename), 'model_best.pth')
            shutil.copyfile(filename, best_path)
            if self.use_wandb and dist.get_rank() == 0:
                wandb.save(best_path)

    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (
                        1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr

    def get_param_groups(self, nowd_keys=()):
        para_groups, para_groups_dbg = {}, {}

        for name, para in self.model.named_parameters():
            if not para.requires_grad:
                continue  # frozen weights
            if len(para.shape) == 1 or name.endswith('.bias') or any(k in name for k in nowd_keys):
                wd_scale, group_name = 0., 'no_decay'
            else:
                wd_scale, group_name = 1., 'decay'

            if group_name not in para_groups:
                para_groups[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
                para_groups_dbg[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
            para_groups[group_name]['params'].append(para)
            para_groups_dbg[group_name]['params'].append(name)

        return list(para_groups.values())

    def vis_reconstruction(self, niters=0):
        import numpy as np
        args = self.args
        loader = self.val_dataloader
        model = self.model

        if dist.get_rank() != 0:
            return

        model.eval()

        with torch.no_grad():
            for batch_data in loader:
                if isinstance(batch_data, list):
                    batch_data = [{k: v.to(args.local_rank) for k, v in crop.items()} for crop in batch_data]
                    image = torch.stack([crop["image"] for crop in batch_data], dim=0).view(-1, 1, 96, 96, 96).to(
                        args.local_rank)
                else:
                    image = batch_data['image'].to(args.local_rank)

                with torch.cuda.amp.autocast(True):
                    _, x, recon, masked_x, pixel_mask = model(image, return_image=True)


                if len(x.shape) == 5:
                    slice_idx = x.shape[2] // 2

                    x_slice = x[:, :, slice_idx]
                    # masked_x_slice = masked_x[:, :, slice_idx]
                    masked_x_slice = pixel_mask[:, :, slice_idx]
                    recon_slice = recon[:, :, slice_idx]

                    x_np = x_slice.cpu().detach().numpy()
                    masked_np = masked_x_slice.cpu().detach().numpy()
                    recon_np = recon_slice.cpu().detach().numpy()

                    import matplotlib.pyplot as plt

                    save_dir = os.path.join(args.save_ckpt_dir, 'visualizations')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f'med_iter{niters:06d}.png')

                    fig, axs = plt.subplots(x_np.shape[0], 3, figsize=(15, 5 * x_np.shape[0]))
                    if x_np.shape[0] == 1:
                        axs = axs.reshape(1, -1)

                    titles = ['raw', 'masked', 'recon']
                    for b in range(x_np.shape[0]):
                        for i, (data, title) in enumerate(zip([x_np[b, 0], masked_np[b, 0], recon_np[b, 0]], titles)):
                            ax = axs[b, i]
                            im = ax.imshow(data, cmap='gray')
                            ax.set_title(f"{title} - Batch {b}")
                            ax.axis('off')

                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150)
                    plt.close()
                    print(f"=> Middle slice visualization saved to: {save_path}")

                    if self.use_wandb:
                        wandb.log(
                            {
                                "vis": wandb.Image(fig)
                            },
                            step=niters,
                        )
                    plt.close(fig)

                elif len(x.shape) == 3:
                    vis_tensor = torch.cat([x, masked_x, recon], dim=0)

                    grid_size = []
                    for pa_size, in_size in zip([args.patch_size] * 3, [96, 96, 96]):
                        grid_size.append(in_size // pa_size)

                    vis_grid_hw = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size,
                                                    in_chans=args.in_chans, hidden_axis='d')
                    print("wandb logging")
                    vis_grid_hw = wandb.Image(vis_grid_hw, caption=f"hw_iter{niters:06d}")
                    # vis_grid_hd = wandb.Image(vis_grid_hd, caption=f"hd_iter{niters:06d}")
                    # vis_grid_wd = wandb.Image(vis_grid_wd, caption=f"wd_iter{niters:06d}")


                    if self.use_wandb:
                        wandb.log(
                            {
                                "vis_hw": vis_grid_hw,
                                # "vis_hd": vis_grid_hd,
                                # "vis_wd": vis_grid_wd
                            },
                            step=niters,
                        )
                    break

                else:
                    print(f"Unsupported output tensor shape: {x.shape}")

                break

    def log_model_stats(self, stats, step=None):
        """Log model statistics to wandb

        Args:
            stats: Dictionary containing various statistics
            step: Current step, uses model's global_step if not provided
        """
        if self.use_wandb and dist.get_rank() == 0:
            try:
                import wandb
                if wandb.run is not None:
                    if step is None and hasattr(self.model, 'module') and hasattr(self.model.module, 'global_step'):
                        step = self.model.module.global_step

                    wandb.log(stats, step=step)
                    if step and step % 100 == 0:
                        print(f"Successfully logged statistics to wandb, step: {step}")
            except Exception as e:
                print(f"Failed to log statistics to wandb: {e}")




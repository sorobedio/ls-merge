"""
Multi-GPU training with DDP.

Launch with:
    torchrun --standalone --nnodes=1 --nproc_per_node=NUM_GPUS train_tf_vae.py [args]
Single-GPU fallback still works:
    python train_tf_vae.py [args]
"""

import argparse
import os
import datetime
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
from tqdm import tqdm
import yaml
import wandb

from zoodatasets.base_datasets import ZooDataset
from utils.util import instantiate_from_config


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description='Autoencoder Training')
    parser.add_argument('--data',          default='modelzoos/zoo_config.yaml',                    type=str)
    parser.add_argument('--dataset',       default='gemma-3-1b-it',              type=str)
    parser.add_argument('--split',         default='train',                        type=str)
    parser.add_argument('--length',        default=640,                          type=int)
    parser.add_argument('--batch_size',    default=128,                            type=int,
                        help='per-GPU batch size')
    parser.add_argument('--num_workers',   default=4,                              type=int,
                        help='dataloader workers per GPU')
    parser.add_argument('--n_epochs',      default=10000,                          type=int)
    parser.add_argument('--warmup_steps',  default=500,                            type=int)
    parser.add_argument('--gradient_clip', default=5.0,                            type=float)
    parser.add_argument('--save_path',     default='vae_checkpoints',              type=str)
    parser.add_argument('-b', '--base',    default='vit_vae/configs/base_vit_vae_config.yaml', type=str)
    parser.add_argument('-n', '--name',    default='vae',                          type=str)
    parser.add_argument('-r', '--resume',  default='',                             type=str)
    parser.add_argument('-s', '--seed',    default=23,                             type=int)
    parser.add_argument('--wandb_project', default='ablat_int',                   type=str)
    parser.add_argument('--wandb_name',    default=None,                           type=str)
    parser.add_argument('--wandb_mode',    default='online',                       type=str,
                        choices=['online', 'offline', 'disabled'])
    parser.add_argument('--log_interval',  default=10,                             type=int)
    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ──────────────────────────────────────────────────────────────────────────────

def setup_distributed():
    """Initialize DDP if torchrun env vars are present, else single-GPU mode."""
    if 'RANK' not in os.environ:
        return False, 0, 1, 0

    rank       = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError('No CUDA device is visible; cannot run GPU training.')

    # Map each process onto a visible GPU. When more processes are launched than
    # there are GPUs (e.g. --nproc_per_node=4 with one visible device), ranks
    # round-robin onto the available GPU(s) so the run still starts.
    device_id = local_rank % num_gpus
    oversubscribed = world_size > num_gpus

    # NCCL assumes one process per GPU and deadlocks when ranks share a device,
    # so fall back to Gloo whenever GPUs are oversubscribed.
    backend = 'gloo' if oversubscribed else 'nccl'

    if rank == 0 and oversubscribed:
        print(f'[warn] {world_size} processes share {num_gpus} GPU(s) '
              f'(round-robin, backend={backend}). This gives no speedup over '
              f'--nproc_per_node={num_gpus} and only exists for debugging.')

    torch.cuda.set_device(device_id)
    dist.init_process_group(backend=backend, init_method='env://')
    dist.barrier()
    return True, rank, world_size, device_id


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def reduce_mean(tensor, world_size):
    """Average a scalar tensor across all ranks."""
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed, rank=0):
    import random
    seed = seed + rank          # different data-augmentation seed per process
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def unwrap(model):
    """Return the underlying module from a DDP wrapper (or the model itself)."""
    return model.module if isinstance(model, DDP) else model


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(model, optimizer, scheduler, n_epochs, train_loader,
          train_sampler, args, rank, world_size):

    if is_main(rank):
        os.makedirs(args.save_path, exist_ok=True)

    device     = next(unwrap(model).parameters()).device
    best_loss  = float('inf')
    global_step = 0
    avg_loss   = float('inf')      # kept for final checkpoint even if 0 epochs

    for epoch in range(n_epochs):

        # DistributedSampler must be re-seeded each epoch for proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = train_rec = train_kl = 0.0
        num_batches = 0

        if is_main(rank):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f'Epoch {epoch + 1}/{n_epochs}')
        else:
            pbar = enumerate(train_loader)

        for batch_idx, inputs in pbar:
            optimizer.zero_grad(set_to_none=True)

            # ── forward ──────────────────────────────────────────────
            x, dec, mu, logvar = model(inputs)

            # padding mask (1 = real weight, 0 = padding); excluded from loss
            mask = inputs.get('mask', None) if isinstance(inputs, dict) else None
            if mask is not None:
                mask = mask.to(x.device)

            loss, logs = unwrap(model).compute_loss(dec, x, mu, logvar, mask=mask)

            if not torch.isfinite(loss):
                if is_main(rank):
                    print(f'  [rank {rank}] Non-finite loss at batch {batch_idx}, skipping.')
                optimizer.zero_grad(set_to_none=True)
                continue

            # ── backward → clip → step ────────────────────────────────
            loss.backward()

            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            optimizer.step()
            scheduler.step()
            global_step += 1

            # ── local metric accumulation ─────────────────────────────
            rec_val = logs['recon_loss']
            kl_val  = logs.get('kl_loss', 0.0)
            rec_val = rec_val.detach().item() if isinstance(rec_val, torch.Tensor) else float(rec_val)
            kl_val  = kl_val.detach().item()  if isinstance(kl_val,  torch.Tensor) else float(kl_val)

            train_loss  += loss.detach().item()
            train_rec   += rec_val
            train_kl    += kl_val
            num_batches += 1

            if is_main(rank):
                pbar.set_postfix({
                    'Loss': f'{train_loss / num_batches:.4f}',
                    'Rec':  f'{train_rec  / num_batches:.4f}',
                    'KL':   f'{train_kl   / num_batches:.4f}',
                    'LR':   f'{optimizer.param_groups[0]["lr"]:.2e}',
                })

        if num_batches == 0:
            if is_main(rank):
                print(f'Epoch {epoch + 1}: all batches skipped.')
            if dist.is_initialized():
                dist.barrier()
            continue

        # ── reduce metrics across GPUs ────────────────────────────────
        denom = max(1, num_batches)
        metrics = torch.tensor(
            [train_loss / denom, train_rec / denom, train_kl / denom],
            device=device
        )
        if dist.is_initialized():
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size

        avg_loss, avg_rec, avg_kl = metrics.tolist()

        # ── logging + checkpointing (main process only) ───────────────
        if is_main(rank):
            print(f'  Avg Loss: {avg_loss:.4f} | Rec: {avg_rec:.4f} | KL: {avg_kl:.4f}')

            wandb.log({
                'epoch':         epoch + 1,
                'train/loss':    avg_loss,
                'train/rec':     avg_rec,
                'train/kl':      avg_kl,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'global_step':   global_step * world_size,
            })

            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f'  → Saving best model (loss={best_loss:.4f})')
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     unwrap(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss':                 best_loss,
                    'world_size':           world_size,
                }, os.path.join(args.save_path, 'best_model.pth'))

            # ── qualitative check ─────────────────────────────────────
            if (epoch + 1) % args.log_interval == 0:
                unwrap(model).eval()
                with torch.no_grad():
                    sample = next(iter(train_loader))
                    x, dec, mu, logvar = unwrap(model)(sample)
                print(f'  Input  first 10: {x[0, :10].cpu().numpy()}')
                print(f'  Output first 10: {dec[0, :10].cpu().numpy()}')
                print(f'  Input  range: [{x.min():.4f}, {x.max():.4f}]')
                print(f'  Output range: [{dec.min():.4f}, {dec.max():.4f}]')
                if mu is not None:
                    print(f'  Mu range:     [{mu.min():.4f}, {mu.max():.4f}]')
                    print(f'  Logvar range: [{logvar.min():.4f}, {logvar.max():.4f}]')
                unwrap(model).train()

        # ── sync all processes at end of epoch ────────────────────────
        if dist.is_initialized():
            dist.barrier()

    # ── final checkpoint (main process only) ─────────────────────────
    if is_main(rank):
        torch.save({
            'epoch':                n_epochs,
            'model_state_dict':     unwrap(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss':                 avg_loss,
            'world_size':           world_size,
        }, os.path.join(args.save_path, 'final_model.pth'))

    return best_loss


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = get_parser()
    args   = parser.parse_args()

    # ── distributed setup ─────────────────────────────────────────────
    distributed, rank, world_size, device_id = setup_distributed()

    seed_everything(args.seed, rank)

    if distributed:
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device_id)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_main(rank):
        print(f'Distributed: {distributed} | World size: {world_size} | Device: {device}')

    # ── wandb (main process only) ──────────────────────────────────────
    if is_main(rank):
        wandb_name = args.wandb_name or \
            f"{args.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=wandb_name,
                   mode=args.wandb_mode, config=vars(args))

    # ── dataset ───────────────────────────────────────────────────────
    if is_main(rank):
        print('Loading dataset...')

    trainset = ZooDataset(
        datapath=args.data, dataset=args.dataset, split=args.split,
        topk=None, scale=0.025, transform=None, normalize=None,
        tgt=None, exd=["bias","norm", "ln"], to_image=False, in_ch=1,
        length=args.length, n_tok=16,
        input_size=32, lamda=0.1,
    )

    if distributed:
        train_sampler = DistributedSampler(
            trainset, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,        # per-GPU
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    if is_main(rank):
        print(f'Dataset: {len(trainset)} samples | '
              f'{len(train_loader)} batches/epoch/GPU | '
              f'Effective batch size: {args.batch_size * world_size}')

    # ── model ─────────────────────────────────────────────────────────
    if is_main(rank):
        print(f'Loading config: {args.base}')

    config = OmegaConf.load(args.base)
    model  = instantiate_from_config(config.model)
    model  = model.to(device)
    model.device = device

    # ── resume (load weights before DDP wrapping) ─────────────────────
    ckpt = None
    if args.resume and os.path.exists(args.resume):
        if is_main(rank):
            print(f'Resuming from: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])

    if is_main(rank):
        print(f'Trainable parameters: {count_trainable_parameters(model):,}')

    # ── optimizer (before DDP so only real params are captured) ────────
    optimizer = model.configure_optimizers()

    # ── wrap with DDP ─────────────────────────────────────────────────
    if distributed:
        model = DDP(model, device_ids=[device_id], output_device=device_id,
                    find_unused_parameters=False)

    # ── scheduler ─────────────────────────────────────────────────────
    total_steps = args.n_epochs * len(train_loader)
    scheduler   = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    # ── restore optimizer / scheduler state ───────────────────────────
    if ckpt is not None:
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    if is_main(rank):
        wandb.config.update({
            'trainable_params':     count_trainable_parameters(unwrap(model)),
            'total_steps':          total_steps,
            'effective_batch_size': args.batch_size * world_size,
        }, allow_val_change=True)

    # ── train ─────────────────────────────────────────────────────────
    if is_main(rank):
        print('Starting training...')

    best_loss = train(
        model, optimizer, scheduler,
        args.n_epochs, train_loader, train_sampler,
        args, rank, world_size
    )

    if is_main(rank):
        print(f'\nTraining complete. Best loss: {best_loss:.4f}')
        wandb.finish()

    cleanup()
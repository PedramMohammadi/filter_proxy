#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .filter_data_loader import FilterDataset
from .model import PreProcessorNet
import random
import torch.optim as optim
import torch.multiprocessing as mp
import os
from torch.utils.data._utils.collate import default_collate
try:
    from google.cloud import storage
except Exception:
    storage = None

# ------------------------------
# Adding random seed to guarantee reproducibility
# ------------------------------
seed = 42  # Or make it an argparse arg
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True # Can pick non-deterministic kernels. Set "torch.backends.cudnn.benchmark = False" and "torch.backends.cudnn.deterministic = True" For absolute deterministic behavior

# ------------------------------
# Helpers
# ------------------------------
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def download_gs_to_local(gs_path, local_dir):
    if not gs_path.startswith("gs://"):
        return gs_path
    if storage is None:
        raise RuntimeError("google-cloud-storage is required for gs://")
    uri = gs_path[5:]
    bucket_name, _, blob_name = uri.partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = str(local_dir / Path(blob_name).name)
    blob.download_to_filename(local_path)
    print(f"Downloaded {gs_path} to {local_path}")
    return local_path

def load_proxy_model(path, device):
    # 1) Prefer TorchScript
    try:
        m = torch.jit.load(path, map_location=device)
        m.eval()
        return m
    except Exception:
        pass
    # 2) Pickled nn.Module
    obj = torch.load(path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj
    # 3) Checkpoint dict — see if it already contains a module
    if isinstance(obj, dict):
        for key in ("model", "module", "net"):
            if key in obj and isinstance(obj[key], torch.nn.Module):
                obj[key].eval()
                return obj[key]
        # Otherwise it’s likely only weights; we can’t build the model without the arch
        known_weight_keys = [k for k in ("state_dict","model_state_dict") if k in obj]
        if known_weight_keys:
            raise RuntimeError(
                f"Proxy checkpoint at {path} contains weights only ({known_weight_keys[0]}). "
                "Provide the proxy architecture and load its state_dict, or export a TorchScript .pt."
            )
    raise RuntimeError(f"Unsupported proxy checkpoint format at {path}")

def _as_scalar_crf(crf_tensor: torch.Tensor) -> torch.Tensor:
    c = crf_tensor.float()
    if c.ndim == 1:
        c = c.unsqueeze(1)                # [B] -> [B,1]
    elif c.ndim >= 2 and c.size(1) != 1:
        c = c.view(c.size(0), -1)[:, :1]  # collapse extras; keep one scalar per sample
    assert c.ndim == 2 and c.size(1) == 1, f"Expected CRF [B,1], got {tuple(c.shape)}"
    return c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training a neural network for pre-processing video frames to enhance VMAF scores.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file containing frame paths and VMAF scores.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory where frames are stored.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model checkpoints, metrics, and plots.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--lambda_l1", type=float, default=0.05, help="Weight for the l1 loss in the training objective.")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay parameter for the optimizer")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Number of maximum consecutive epochs without validation loss improvement to stop training.")
    parser.add_argument("--use_plateau_scheduler", action="store_true", help="Modify the learning rate when learning plateaus.")
    parser.add_argument("--crop_size", type=int, default=128, help="Size of square crop patches from frames.")
    parser.add_argument("--n_frames", type=int, default=3, help="Number of consecutive frames per sample.")
    parser.add_argument("--early_stop", action="store_true", help="Enable early stopping based on validation loss.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from.")
    args = parser.parse_args()

    if args.n_frames % 2 == 0:
        raise ValueError(f"--n_frames must be odd (got {args.n_frames})")

    # Everything is local (prefetched with gsutil rsync to /data/dataset)
    def _gs_to_local(p: str, local_root="/data"):
        if not isinstance(p, str) or not p.startswith("gs://"):
            return p
        # gs://<bucket>/<suffix>  → /data/<suffix>
        bucket_and_suffix = p[5:]
        _, _, suffix = bucket_and_suffix.partition("/")
        return f"{local_root}/{suffix}".rstrip("/")

    # Redirect any gs:// paths to the local mirror
    args.root_dir = _gs_to_local(args.root_dir, "/data")

    mp.set_start_method('spawn', force=True)

    train_dataset = FilterDataset(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        crop_size=args.crop_size,
        fixed_crop=False,
        n_frames=args.n_frames,
        split="Training"
    )
    val_dataset = FilterDataset(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        crop_size=args.crop_size,
        fixed_crop=True,
        n_frames=args.n_frames,
        split="Validation"
    )

    train_loader = TorchDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=16,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
        worker_init_fn=worker_init_fn, collate_fn=collate_skip_none
    )
    val_loader = TorchDataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, prefetch_factor=16,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
        worker_init_fn=worker_init_fn, collate_fn=collate_skip_none
    )

    model = PreProcessorNet(
        n_frames=args.n_frames,
        crop_size=args.crop_size,
    ).to(args.device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.use_plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=True
        )
    else:
        scheduler = None
    
    start_epoch = 0
    best_val_loss = float('inf')
    early_stop_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_path = args.resume

        # If a directory was passed, try the file we actually save
        if os.path.isdir(resume_path):
            resume_path = str(Path(resume_path) / "filter_model.pth")
        
        # If it's a GCS URI, download it locally first
        if isinstance(resume_path, str) and resume_path.startswith("gs://"):
            if storage is None:
                raise RuntimeError("google-cloud-storage is required to resume from gs://")
            uri = resume_path[len("gs://"):]
            bucket_name, _, blob_name = uri.partition("/")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            local_dir = Path(args.output_dir) if args.output_dir else Path("/tmp/checkpoints")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_resume = str(local_dir / "resume_checkpoint.pth")
            blob.download_to_filename(local_resume)
            print(f"Downloaded resume checkpoint from {args.resume} to {local_resume}")
            resume_path = local_resume
        
        if os.path.isfile(resume_path):
            print(f"Resuming from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=args.device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.use_plateau_scheduler and checkpoint.get('scheduler') is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            #if you want to override LR after resume:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
        else:
            print(f"WARNING: --resume specified but file not found: {resume_path}")
        
    
    vmaf_proxy_gs    = "gs://vmaf_proxy_training_checkpoints/checkpoints/vmaf_proxy.pt"
    encoder_proxy_gs = "gs://encoder_proxy_training_checkpoints/checkpoints/encoder_proxy.pt"

    proxies_dir = output_dir / "proxies"
    vmaf_proxy_local = download_gs_to_local(vmaf_proxy_gs, proxies_dir)
    encoder_proxy_local = download_gs_to_local(encoder_proxy_gs, proxies_dir)

    #Loading encoder and VMAF proxy models
    vmaf_proxy    = load_proxy_model(vmaf_proxy_local, args.device)
    encoder_proxy = load_proxy_model(encoder_proxy_local, args.device)

    for m in (vmaf_proxy, encoder_proxy):
        for p in m.parameters():
            p.requires_grad_(False)

    # Training loop
    train_losses = []
    val_losses = []
    val_vmaf_gains = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        num_train_batches = 0

        for batch in tqdm(train_loader, disable=True):

            if batch is None:
                continue
            
            x_ref, x_dist, y, crf = batch
            x_ref, x_dist, y, crf = x_ref.to(args.device), x_dist.to(args.device), y.to(args.device), crf.to(args.device)
            crf = _as_scalar_crf(crf)

            optimizer.zero_grad()
            
            # Indices for center frame
            n = x_ref.shape[1]
            i = n // 2
            x_ref_center = x_ref[:, i:i+1]  # [B,1,H,W]

            # ----- Baseline distorted group (orig) via encoder proxy -----
            dec_tminus = encoder_proxy(x_ref[:, i-1:i], crf)  # t-1
            dec_t      = encoder_proxy(x_ref_center, crf)     # t
            dec_tplus  = encoder_proxy(x_ref[:, i+1:i+2], crf)  # t+1

            # ----- Preprocess center, then build preprocessed distorted group -----
            preproc_center = model(x_ref)                     # [B,1,H,W] (model uses 3-frame context)
            dec_t_edit     = encoder_proxy(preproc_center, crf)
            # Keep neighbors unchanged; swapping only the center reflects the edit
            x_dist_pre = torch.cat([dec_tminus.detach(), dec_t_edit, dec_tplus.detach()], dim=1)

            # VMAF after preprocessing (group vs group)
            vmaf_preproc = torch.clamp(vmaf_proxy(x_ref, x_dist_pre), 0.0, 1.0)

            # Regularize toward identity on center frame
            reg_loss = criterion(preproc_center, x_ref_center)

            # Maximize VMAF after preprocessing
            loss = -vmaf_preproc.mean() + args.lambda_l1 * reg_loss
            
            if not torch.isfinite(loss):
                print(f"Non-finite loss detected: {loss.item()}. Skipping batch.")
                continue
            
            loss.backward()
            
            # First clip individual values, then compute norm
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gnorm):
                # Only step if gradients are finite
                optimizer.step()
                train_loss += loss.item()
                num_train_batches += 1
            else:
                print(f"non-finite grad norm: {gnorm}; skipping step.")
                lo, hi = preproc_center.min().item(), preproc_center.max().item()
                print(f"Loss: {loss.item():.6f}, Preproc_center range: [{lo:.6f}, {hi:.6f}]")

        if num_train_batches == 0:
            print("[WARN] No valid training batches in this epoch; skipping metrics.")
            train_losses.append(float('inf'))  # Indicate invalid epoch
            continue
        avg_train_loss = train_loss / max(1, num_train_batches)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_vmaf_orig_total = 0.0
        val_vmaf_preproc_total = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, disable=True):
                
                if batch is None:
                    continue
                
                x_ref, x_dist, y, crf = batch
                x_ref, x_dist, y, crf = x_ref.to(args.device), x_dist.to(args.device), y.to(args.device), crf.to(args.device)
                crf = _as_scalar_crf(crf)
                
                # Skip batch if any tensor has NaN/Inf
                if not (torch.isfinite(x_ref).all() and torch.isfinite(x_dist).all() and torch.isfinite(y).all() and torch.isfinite(crf).all()):
                    continue

                # Indices for center frame
                n = x_ref.shape[1]
                i = n // 2
                x_ref_center = x_ref[:, i:i+1]  # [B,1,H,W]

                # ----- Baseline distorted group (orig) -----
                dec_tminus = encoder_proxy(x_ref[:, i-1:i], crf)
                dec_t      = encoder_proxy(x_ref_center, crf)
                dec_tplus  = encoder_proxy(x_ref[:, i+1:i+2], crf)
                x_dist_orig = torch.cat([dec_tminus, dec_t, dec_tplus], dim=1)
                vmaf_orig = torch.clamp(vmaf_proxy(x_ref, x_dist_orig), 0.0, 1.0).mean()

                # ----- Preprocessed distorted group -----
                preproc_center = model(x_ref)                     # [B,1,H,W]
                dec_t_edit     = encoder_proxy(preproc_center, crf)
                x_dist_pre     = torch.cat([dec_tminus, dec_t_edit, dec_tplus], dim=1)

                vmaf_preproc = torch.clamp(vmaf_proxy(x_ref, x_dist_pre), 0.0, 1.0)
                reg_loss     = criterion(preproc_center, x_ref_center)

                loss = -vmaf_preproc.mean() + args.lambda_l1 * reg_loss
                val_loss += loss.item()
                val_vmaf_orig_total    += vmaf_orig.item()
                val_vmaf_preproc_total += vmaf_preproc.mean().item()
                
                num_val_batches += 1
        
        if num_val_batches == 0:
            print("[WARN] No valid validation batches this epoch; skipping metrics/ckpt.")
            val_losses.append(float('inf'))
            val_vmaf_gains.append(0.0)
            early_stop_counter += 1
            print(f"No improvement in val_loss for {early_stop_counter} epoch(s).")
            # "very bad" val loss
            if scheduler:
                scheduler.step(float('inf'))  # forces "no improvement" this epoch
            if args.early_stop and early_stop_counter >= args.early_stop_patience:
                print("Early stopping triggered.")
                break
            continue

        avg_val_loss = val_loss / max(1, num_val_batches)
        avg_vmaf_gain = (val_vmaf_preproc_total - val_vmaf_orig_total) / num_val_batches
        val_losses.append(avg_val_loss)
        val_vmaf_gains.append(avg_vmaf_gain)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val VMAF Gain = {avg_vmaf_gain:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, output_dir / "filter_model.pth")

            if storage is not None:
                client = storage.Client()
                bucket = client.bucket('vmaf_proxy_training_checkpoints')
                blob = bucket.blob('checkpoints/filter_model.pth')
                blob.upload_from_filename(str(output_dir / "filter_model.pth"))
                print("Checkpoint uploaded to GCS")
            else:
                print("Skipping GCS upload (google-cloud-storage not available).")

            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement in val_loss for {early_stop_counter} epoch(s).")

        if scheduler:
            scheduler.step(avg_val_loss)
        
        if args.early_stop and early_stop_counter >= args.early_stop_patience:
            print("Early stopping triggered.")
            break
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(output_dir / 'loss_curve.png')
    plt.close()

    # Plot VMAF gains
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_vmaf_gains) + 1), val_vmaf_gains, label='Val VMAF Gain')
    plt.xlabel('Epoch')
    plt.ylabel('VMAF Gain')
    plt.legend()
    plt.title('Validation VMAF Gain')
    plt.savefig(output_dir / 'vmaf_gain_curve.png')
    plt.close()
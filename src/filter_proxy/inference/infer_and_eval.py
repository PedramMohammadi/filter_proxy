#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader as TorchDataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Import your model and dataset classes
from .filter_data_loader import FilterDataset
from .model import PreProcessorNet

try:
    from google.cloud import storage
except Exception:
    storage = None


def download_gs_to_local(gs_path, local_dir):
    """Download a file from GCS to local directory."""
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
    """Load a proxy model (VMAF or encoder) from checkpoint."""
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
    # 3) Checkpoint dict
    if isinstance(obj, dict):
        for key in ("model", "module", "net"):
            if key in obj and isinstance(obj[key], torch.nn.Module):
                obj[key].eval()
                return obj[key]
        known_weight_keys = [k for k in ("state_dict", "model_state_dict") if k in obj]
        if known_weight_keys:
            raise RuntimeError(
                f"Proxy checkpoint at {path} contains weights only ({known_weight_keys[0]}). "
                "Provide the proxy architecture and load its state_dict, or export a TorchScript .pt."
            )
    raise RuntimeError(f"Unsupported proxy checkpoint format at {path}")


def load_preprocessor_model(checkpoint_path, n_frames, crop_size, device):
    """Load the trained PreProcessorNet model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = PreProcessorNet(n_frames=n_frames, crop_size=crop_size).to(device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the dict itself is the state dict
            model.load_state_dict(checkpoint)
    else:
        raise RuntimeError(f"Unexpected checkpoint format at {checkpoint_path}")
    
    model.eval()
    return model


def _as_scalar_crf(crf_tensor: torch.Tensor) -> torch.Tensor:
    """Ensure CRF tensor is [B, 1] shape."""
    c = crf_tensor.float()
    if c.ndim == 1:
        c = c.unsqueeze(1)
    elif c.ndim >= 2 and c.size(1) != 1:
        c = c.view(c.size(0), -1)[:, :1]
    assert c.ndim == 2 and c.size(1) == 1, f"Expected CRF [B,1], got {tuple(c.shape)}"
    return c


def collate_skip_none(batch):
    """Collate function that skips None samples."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


def compute_metrics(results):
    """Compute aggregate metrics from results."""
    vmaf_orig = np.array([r['vmaf_orig'] for r in results])
    vmaf_preproc = np.array([r['vmaf_preproc'] for r in results])
    vmaf_gains = vmaf_preproc - vmaf_orig
    
    metrics = {
        'mean_vmaf_orig': float(np.mean(vmaf_orig)),
        'mean_vmaf_preproc': float(np.mean(vmaf_preproc)),
        'mean_vmaf_gain': float(np.mean(vmaf_gains)),
        'median_vmaf_gain': float(np.median(vmaf_gains)),
        'std_vmaf_gain': float(np.std(vmaf_gains)),
        'min_vmaf_gain': float(np.min(vmaf_gains)),
        'max_vmaf_gain': float(np.max(vmaf_gains)),
        'pct_improved': float(np.mean(vmaf_gains > 0) * 100),
        'pct_degraded': float(np.mean(vmaf_gains < 0) * 100),
    }
    
    # Compute by CRF if available
    if 'crf' in results[0]:
        crf_metrics = defaultdict(list)
        for r in results:
            crf_val = r['crf']
            crf_metrics[crf_val].append(r['vmaf_preproc'] - r['vmaf_orig'])
        
        metrics['by_crf'] = {}
        for crf, gains in sorted(crf_metrics.items()):
            metrics['by_crf'][f'crf_{crf}'] = {
                'mean_gain': float(np.mean(gains)),
                'median_gain': float(np.median(gains)),
                'count': len(gains)
            }
    
    return metrics


def plot_results(results, output_dir):
    """Generate visualization plots for inference results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vmaf_orig = np.array([r['vmaf_orig'] for r in results])
    vmaf_preproc = np.array([r['vmaf_preproc'] for r in results])
    vmaf_gains = vmaf_preproc - vmaf_orig
    
    # Plot 1: VMAF gain distribution
    plt.figure(figsize=(10, 6))
    plt.hist(vmaf_gains, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(vmaf_gains), color='red', linestyle='--', 
                label=f'Mean: {np.mean(vmaf_gains):.3f}')
    plt.axvline(np.median(vmaf_gains), color='green', linestyle='--', 
                label=f'Median: {np.median(vmaf_gains):.3f}')
    plt.xlabel('VMAF Gain')
    plt.ylabel('Frequency')
    plt.title('Distribution of VMAF Gains')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'vmaf_gain_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Before vs After scatter
    plt.figure(figsize=(10, 10))
    plt.scatter(vmaf_orig, vmaf_preproc, alpha=0.5, s=10)
    plt.plot([0, 100], [0, 100], 'r--', label='No change')
    plt.xlabel('Original VMAF')
    plt.ylabel('Preprocessed VMAF')
    plt.title('VMAF: Original vs Preprocessed')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim([min(vmaf_orig.min(), vmaf_preproc.min()) - 5, 
              max(vmaf_orig.max(), vmaf_preproc.max()) + 5])
    plt.ylim([min(vmaf_orig.min(), vmaf_preproc.min()) - 5, 
              max(vmaf_orig.max(), vmaf_preproc.max()) + 5])
    plt.savefig(output_dir / 'vmaf_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: VMAF gain by CRF (if available)
    if 'crf' in results[0]:
        crf_values = np.array([r['crf'] for r in results])
        unique_crfs = sorted(np.unique(crf_values))
        
        mean_gains = []
        std_gains = []
        for crf in unique_crfs:
            mask = crf_values == crf
            gains = vmaf_gains[mask]
            mean_gains.append(np.mean(gains))
            std_gains.append(np.std(gains))
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(unique_crfs, mean_gains, yerr=std_gains, 
                    fmt='o-', capsize=5, capthick=2)
        plt.xlabel('CRF Value')
        plt.ylabel('Mean VMAF Gain')
        plt.title('VMAF Gain by CRF Value')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'vmaf_gain_by_crf.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 4: Cumulative distribution
    plt.figure(figsize=(10, 6))
    sorted_gains = np.sort(vmaf_gains)
    cumulative = np.arange(1, len(sorted_gains) + 1) / len(sorted_gains) * 100
    plt.plot(sorted_gains, cumulative, linewidth=2)
    plt.axvline(0, color='red', linestyle='--', label='Zero gain')
    plt.xlabel('VMAF Gain')
    plt.ylabel('Cumulative Percentage (%)')
    plt.title('Cumulative Distribution of VMAF Gains')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'vmaf_gain_cumulative.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def run_inference(args):
    """Run inference and evaluation on the test dataset."""
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle GCS paths
    def _gs_to_local(p: str, local_root="/data"):
        if not isinstance(p, str) or not p.startswith("gs://"):
            return p
        bucket_and_suffix = p[5:]
        _, _, suffix = bucket_and_suffix.partition("/")
        return f"{local_root}/{suffix}".rstrip("/")
    
    # Redirect GCS paths to local mirror if needed
    if args.root_dir.startswith("gs://"):
        args.root_dir = _gs_to_local(args.root_dir, "/data")
    
    # Download models from GCS
    proxies_dir = output_dir / "proxies"
    proxies_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading models from GCS...")
    preprocessor_local = download_gs_to_local(args.model_checkpoint, proxies_dir)
    vmaf_proxy_local = download_gs_to_local(args.vmaf_proxy, proxies_dir)
    encoder_proxy_local = download_gs_to_local(args.encoder_proxy, proxies_dir)
    
    # Load models
    print("Loading models...")
    device = torch.device(args.device)
    
    preprocessor = load_preprocessor_model(
        preprocessor_local, 
        args.n_frames, 
        args.crop_size, 
        device
    )
    vmaf_proxy = load_proxy_model(vmaf_proxy_local, device)
    encoder_proxy = load_proxy_model(encoder_proxy_local, device)
    
    # Freeze proxy models
    for m in (vmaf_proxy, encoder_proxy):
        for p in m.parameters():
            p.requires_grad_(False)
    
    print("Models loaded successfully.")
    
    # Setup dataset
    print(f"Loading dataset from {args.csv_file}...")
    test_dataset = FilterDataset(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        crop_size=args.crop_size,
        fixed_crop=True,  # Use fixed crop for consistent evaluation
        n_frames=args.n_frames,
        split="Testing"
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        collate_fn=collate_skip_none
    )
    
    print(f"Dataset loaded: {len(test_dataset)} samples")
    
    # Run inference
    print("Running inference...")
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if batch is None:
                continue
            
            x_ref, x_dist, y_true, crf = batch
            x_ref = x_ref.to(device)
            x_dist = x_dist.to(device)
            y_true = y_true.to(device)
            crf = _as_scalar_crf(crf.to(device))
            
            # Get center frame index
            n = x_ref.shape[1]
            i = n // 2
            x_ref_center = x_ref[:, i:i+1]
            
            # === Original pipeline (no preprocessing) ===
            dec_tminus_orig = encoder_proxy(x_ref[:, i-1:i], crf)
            dec_t_orig = encoder_proxy(x_ref_center, crf)
            dec_tplus_orig = encoder_proxy(x_ref[:, i+1:i+2], crf)
            x_dist_orig = torch.cat([dec_tminus_orig, dec_t_orig, dec_tplus_orig], dim=1)
            vmaf_orig = torch.clamp(vmaf_proxy(x_ref, x_dist_orig), 0.0, 1.0)
            
            # === Preprocessed pipeline ===
            preproc_center = preprocessor(x_ref)
            dec_t_preproc = encoder_proxy(preproc_center, crf)
            x_dist_preproc = torch.cat([dec_tminus_orig, dec_t_preproc, dec_tplus_orig], dim=1)
            vmaf_preproc = torch.clamp(vmaf_proxy(x_ref, x_dist_preproc), 0.0, 1.0)
            
            # Store results (convert to 0-100 scale)
            for j in range(x_ref.size(0)):
                results.append({
                    'batch_idx': batch_idx,
                    'sample_idx': batch_idx * args.batch_size + j,
                    'vmaf_orig': vmaf_orig[j].item() * 100,
                    'vmaf_preproc': vmaf_preproc[j].item() * 100,
                    'vmaf_ground_truth': y_true[j].item() * 100,
                    'crf': (crf[j].item() * 51.0),  # Convert back to original scale
                })
            
            # Optional: save some preprocessed frames
            if args.save_frames and batch_idx < args.num_frames_to_save:
                frames_dir = output_dir / "preprocessed_frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                
                for j in range(min(x_ref.size(0), 4)):  # Save first 4 in batch
                    idx = batch_idx * args.batch_size + j
                    
                    # Save reference
                    ref_img = x_ref_center[j, 0].cpu().numpy()
                    plt.imsave(
                        frames_dir / f"sample_{idx:04d}_reference.png",
                        ref_img, cmap='gray'
                    )
                    
                    # Save preprocessed
                    preproc_img = preproc_center[j, 0].cpu().numpy()
                    plt.imsave(
                        frames_dir / f"sample_{idx:04d}_preprocessed.png",
                        preproc_img, cmap='gray'
                    )
                    
                    # Save difference map
                    diff = np.abs(preproc_img - ref_img)
                    plt.imsave(
                        frames_dir / f"sample_{idx:04d}_difference.png",
                        diff, cmap='hot'
                    )
    
    print(f"Inference complete. Processed {len(results)} samples.")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(results)
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE RESULTS SUMMARY")
    print("="*60)
    print(f"Total samples: {len(results)}")
    print(f"Mean VMAF (original): {metrics['mean_vmaf_orig']:.2f}")
    print(f"Mean VMAF (preprocessed): {metrics['mean_vmaf_preproc']:.2f}")
    print(f"Mean VMAF gain: {metrics['mean_vmaf_gain']:.3f}")
    print(f"Median VMAF gain: {metrics['median_vmaf_gain']:.3f}")
    print(f"Std VMAF gain: {metrics['std_vmaf_gain']:.3f}")
    print(f"Min/Max VMAF gain: {metrics['min_vmaf_gain']:.3f} / {metrics['max_vmaf_gain']:.3f}")
    print(f"% samples improved: {metrics['pct_improved']:.1f}%")
    print(f"% samples degraded: {metrics['pct_degraded']:.1f}%")
    
    if 'by_crf' in metrics:
        print("\nResults by CRF:")
        for crf_key, crf_metrics in sorted(metrics['by_crf'].items()):
            print(f"  {crf_key}: mean gain = {crf_metrics['mean_gain']:.3f}, "
                  f"median = {crf_metrics['median_gain']:.3f}, "
                  f"n = {crf_metrics['count']}")
    print("="*60 + "\n")
    
    # Save results
    print("Saving results...")
    
    # Save detailed results as JSON
    with open(output_dir / 'inference_results.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'samples': results
        }, f, indent=2)
    
    # Save as CSV for easy analysis
    df = pd.DataFrame(results)
    df['vmaf_gain'] = df['vmaf_preproc'] - df['vmaf_orig']
    df.to_csv(output_dir / 'inference_results.csv', index=False)
    
    # Generate plots
    print("Generating plots...")
    plot_results(results, output_dir)
    
    print(f"\nAll results saved to {output_dir}")
    print(f"  - inference_results.json")
    print(f"  - inference_results.csv")
    print(f"  - vmaf_gain_distribution.png")
    print(f"  - vmaf_scatter.png")
    print(f"  - vmaf_gain_by_crf.png")
    print(f"  - vmaf_gain_cumulative.png")
    
    return metrics, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and evaluation script for video pre-processing model")
    
    # Model paths
    parser.add_argument("--model_checkpoint", type=str, default="gs://vmaf_proxy_training_checkpoints/checkpoints/filter_model.pth", help="Path to trained preprocessor model checkpoint (local or gs://)")
    parser.add_argument("--vmaf_proxy", type=str, default="gs://vmaf_proxy_training_checkpoints/checkpoints/vmaf_proxy.pt", help="Path to VMAF proxy model")
    parser.add_argument("--encoder_proxy", type=str, default="gs://encoder_proxy_training_checkpoints/checkpoints/encoder_proxy.pt", help="Path to encoder proxy model")
    
    # Dataset paths
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file containing test data")
    parser.add_argument("--root_dir", type=str, default="gs://vmaf_proxy_dataset/dataset/Frames/", help="Root directory where frames are stored")
    
    # Model parameters (must match training)
    parser.add_argument("--n_frames", type=int, default=3, help="Number of frames (must match training)")
    parser.add_argument("--crop_size", type=int, default=128, help="Crop size (must match training)")
    
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Directory to save inference results and plots")
    parser.add_argument("--save_frames", action="store_true", help="Save some preprocessed frames for visual inspection")
    parser.add_argument("--num_frames_to_save", type=int, default=10, help="Number of batches to save frames from (if --save_frames)")
    
    args = parser.parse_args()
    
    # Validate
    if args.n_frames % 2 == 0:
        raise ValueError(f"--n_frames must be odd (got {args.n_frames})")
    
    # Run inference
    metrics, results = run_inference(args)
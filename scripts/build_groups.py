#!/usr/bin/env python3

import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# CLI Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(description="Extract grayscale frames and group them for proxy VMAF training.")
parser.add_argument("--n_frames", type=int, default=3, help="Number of frames per group (must be odd)")
parser.add_argument("--vmaf_csvs", nargs="+", required=True, help="One or more CSV files to combine")
parser.add_argument("--output_csv", type=str, default="proxy_vmaf_groups.csv", help="Output CSV file with grouped paths")
parser.add_argument("--enable_balancing", action="store_true", help="Enable sampling across VMAF")
parser.add_argument("--clips_per_bin", type=int, default=40, help="Clips per VMAF bin")
parser.add_argument("--num_bins", type=int, default=10, help="Number of VMAF bins across 0–100")
parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

# ------------------------------
# VMAF Quality Balancing
# ------------------------------
def apply_vmaf_balancing(df, clips_per_bin, num_bins, seed):
    print("Balancing VMAF distribution across clips...")
    df["clip_id"] = df["sequence_name"] + "_" + "_crf" + df["crf"].astype(str)
    avg_df = df.groupby("clip_id")["vmaf_score"].mean().reset_index()
    avg_df.rename(columns={"vmaf_score": "average_vmaf"}, inplace=True)

    # Bin edges and assignment
    bins = np.linspace(0, 100, num_bins + 1)
    avg_df["bin"] = pd.cut(avg_df["average_vmaf"], bins=bins, labels=False, include_lowest=True)
    avg_df["crf"] = avg_df["clip_id"].str.extract(r"_crf(\d+)$").astype(int)

    # Plot histogram BEFORE sampling
    plt.figure(figsize=(10, 5))
    plt.hist(avg_df["average_vmaf"], bins=bins, edgecolor='black', alpha=0.5, label="Before Sampling")

    # Sample from each bin
    sampled_dfs = []
    for b in range(num_bins):
        bin_group = avg_df[avg_df["bin"] == b]
        bin_size = len(bin_group)

        if bin_size == 0:
            continue
        elif bin_size <= clips_per_bin:
            print(f"\nBin {b} only has {bin_size} clips but {clips_per_bin} were requested. Including all.")
            sampled = bin_group
        else:
            sampled = bin_group.sample(n=clips_per_bin, random_state=seed)

        sampled_dfs.append(sampled)

    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    sampled_df["bin"] = pd.cut(sampled_df["average_vmaf"], bins=bins, labels=False, include_lowest=True)

    # Histogram AFTER sampling
    plt.hist(sampled_df["average_vmaf"], bins=bins, edgecolor='black', alpha=0.7, label="After Sampling")

    plt.title("Histogram of Average VMAF per Clip (Before and After Sampling)")
    plt.xlabel("VMAF Score")
    plt.ylabel("Number of Clips")
    plt.grid(True)
    plt.legend()
    plt.xticks(bins, rotation=45)
    plt.tight_layout()
    plt.savefig("vmaf_distribution.png")
    print("\nSaved histogram to vmaf_distribution.png")

    # Final filtering for full dataset
    df["clip_id"] = df["sequence_name"] + "_" + "_crf" + df["crf"].astype(str)
    filtered = df[df["clip_id"].isin(sampled_df["clip_id"])].copy()
    print(f"\nSelected {filtered['clip_id'].nunique()} clips across {num_bins} bins.")
    return filtered

# ------------------------------
# Build group CSV
# ------------------------------
df_list = [pd.read_csv(csv) for csv in args.vmaf_csvs]
df = pd.concat(df_list, ignore_index=True)
df_train = df[df["split"] == "Training"].copy()
df_rest  = df[df["split"] != "Training"].copy()
if args.enable_balancing:
    df_train = apply_vmaf_balancing(df_train, args.clips_per_bin, args.num_bins, args.random_seed)
df = pd.concat([df_train, df_rest], ignore_index=True)
rows = []
half = args.n_frames // 2
if args.n_frames % 2 == 0:
    raise ValueError("n_frames must be odd")

for (name, crf), group_df in df.groupby(["sequence_name", "crf"]):
    group_df = group_df.sort_values("frame_num").reset_index(drop=True)
    for i in range(half, len(group_df) - half):
        row = group_df.loc[i]
        split = row["split"]  # default; update if split info is in VMAF CSV

        dist_base = Path("Frames") / "Compressed" / split / name / f"crf{crf}"
        ref_base = Path("Frames") / "Original" / split / name

        row_dict = {
            "split": split,
            "sequence_name": name,
            "crf": crf,
            "frame_num": row["frame_num"],
            "vmaf_score": row["vmaf_score"]
        }
        for offset in range(-half, half + 1):
            fnum = int(group_df.loc[i + offset]['frame_num'])
            suffix = f"t{offset:+d}".replace("+0", "")
            row_dict[f"dist_frame_{suffix}_path"] = str(dist_base / f"frame_{fnum:04d}.png")
            row_dict[f"ref_frame_{suffix}_path"] = str(ref_base / f"frame_{fnum:04d}.png")
        rows.append(row_dict)

out_df = pd.DataFrame(rows)
out_df.to_csv(args.output_csv, index=False)
print(f"Frame group CSV saved to: {args.output_csv}")

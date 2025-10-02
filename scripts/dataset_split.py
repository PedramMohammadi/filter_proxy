#!/usr/bin/env python3

import os
import random
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Split UGC dataset into train, validation, and test sets.")
    parser.add_argument('--source_dir', required=True, help='Directory containing the original dataset')
    parser.add_argument('--train_dir', required=True, help='Directory to store training set')
    parser.add_argument('--val_dir', required=True, help='Directory to store validation set')
    parser.add_argument('--test_dir', required=True, help='Directory to store test set')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test data')
    parser.add_argument('--input_formats', nargs='+', default=['.mkv', '.mp4', '.webm', '.mov', '.yuv'], help='Video file extensions to include (e.g., .mkv .yuv .mp4)')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffling")
    return parser.parse_args()

def copy_files(file_list, source_dir, destination):
    os.makedirs(destination, exist_ok=True)
    for filename in file_list:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(destination, filename)
        shutil.copy2(src, dst)
        print(f"Copied: {filename} ➜ {destination}")

def main():
    args = parse_args()

    if not os.path.exists(args.source_dir):
        raise FileNotFoundError(f"Source directory does not exist: {args.source_dir}")

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"Train/Val/Test ratios must sum to 1.0 (got {total_ratio})")

    # Collect files matching provided input formats
    video_files = [
        f for f in os.listdir(args.source_dir)
        if any(f.lower().endswith(ext.lower()) for ext in args.input_formats) and os.path.isfile(os.path.join(args.source_dir, f))
    ]

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(video_files)
    total = len(video_files)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    train_files = video_files[:train_end]
    val_files = video_files[train_end:val_end]
    test_files = video_files[val_end:]

    print(f"Total files: {total}")
    print(f"Training: {len(train_files)} | Validation: {len(val_files)} | Testing: {len(test_files)}")

    # Copy files
    copy_files(train_files, args.source_dir, args.train_dir)
    copy_files(val_files, args.source_dir, args.val_dir)
    copy_files(test_files, args.source_dir, args.test_dir)

if __name__ == "__main__":
    main()

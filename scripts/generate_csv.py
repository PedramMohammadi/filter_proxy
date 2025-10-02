#!/usr/bin/env python3

import argparse
import json
import csv
from pathlib import Path
import re
from collections import defaultdict

parser = argparse.ArgumentParser(description="Parse VMAF JSON logs and generate separate CSVs per split")
parser.add_argument("--log_dir", type=str, required=True, help="Directory with VMAF JSON logs")
parser.add_argument("--csv_root", type=str, default="vmaf_frame_level", help="Root path for CSV outputs (no .csv)")
args = parser.parse_args()

json_paths = list(Path(args.log_dir).rglob("*_crf*_vmaf.json"))
print(f"Found {len(json_paths)} VMAF JSON log files")

split_files = {}
split_writers = {}
split_counts = defaultdict(int)
malformed_files = []

def get_writer(split):
    if split not in split_writers:
        out_path = f"{args.csv_root}_VMAF_data_{split}.csv"
        f = open(out_path, "w", newline="")
        w = csv.writer(f)
        w.writerow(["split", "sequence_name", "crf", "frame_num", "vmaf_score"])
        split_files[split] = f
        split_writers[split] = w
    return split_writers[split]

for json_path in json_paths:
    try:
        rel_parts = json_path.relative_to(args.log_dir).parts
        split = rel_parts[0] if len(rel_parts) > 1 else "Unknown"
    except Exception as e:
        print(f"Could not determine split for {json_path}: {e}")
        continue

    filename = json_path.stem
    match = re.match(r"(.+)_crf(\d+)_vmaf", filename)
    if not match:
        malformed_files.append(filename)
        continue

    sequence_name, crf = match.groups()
    split_counts[split] += 1

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            frames = data.get("frames", [])
            if not frames:
                print(f"No frames found in {json_path}")
                continue

            writer = get_writer(split)
            for frame in frames:
                writer.writerow([split, sequence_name, crf, frame["frameNum"], frame["metrics"]["vmaf"]])
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        continue

# Close all CSVs
for f in split_files.values():
    f.close()

print(f"All CSVs written.")
print(f"Split summary:")
for split, count in split_counts.items():
    print(f"  - {split}: {count} files")

if malformed_files:
    print(f"Skipped {len(malformed_files)} files due to malformed names:")
    for f in malformed_files:
        print(f"   - {f}")

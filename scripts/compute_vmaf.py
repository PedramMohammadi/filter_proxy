#!/usr/bin/env python3

import argparse
import subprocess
import json
from pathlib import Path
import os
import re

# ------------------------------
# CLI Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(description="Compute VMAF scores between reference and compressed video sequences.")
parser.add_argument("--ref_dir", type=str, required=True, help="Base directory containing reference videos (e.g. Training/Validation/Testing)")
parser.add_argument("--dist_dir", type=str, required=True, help="Base directory containing compressed videos (same structure)")
parser.add_argument("--log_dir", type=str, required=True, help="Directory to store VMAF JSON logs")
parser.add_argument("--width", type=int, help="Width of reference video (required for .yuv)")
parser.add_argument("--height", type=int, help="Height of reference video (required for .yuv)")
parser.add_argument("--fps", type=int, help="Frame rate of reference video (required for .yuv)")
parser.add_argument("--input_formats", nargs="+", default=["yuv", "mp4", "mkv", "y4m"], help="Accepted reference formats (default: yuv, mp4, mkv, y4m)")
parser.add_argument("--ffmpeg_dir", type=str, default="", help="Optional directory containing ffmpeg and ffprobe binaries")
args = parser.parse_args()

# ------------------------------
# Determine binary paths
# ------------------------------
ffmpeg_bin = os.path.join(args.ffmpeg_dir, "ffmpeg") if args.ffmpeg_dir else "ffmpeg"
ffprobe_bin = os.path.join(args.ffmpeg_dir, "ffprobe") if args.ffmpeg_dir else "ffprobe"

Path(args.log_dir).mkdir(parents=True, exist_ok=True)

# ------------------------------
# Helper: Extract metadata using ffprobe
# ------------------------------
def get_metadata_with_ffprobe(filepath):
    try:
        cmd = [
            ffprobe_bin, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json", str(filepath)
        ]
        output = subprocess.check_output(cmd).decode()
        info = json.loads(output)['streams'][0]
        width = info['width']
        height = info['height']
        fps = eval(info['r_frame_rate'])  # Handles "30/1" etc.
        return str(width), str(height), str(int(fps))
    except Exception as e:
        print(f"Could not extract metadata from {filepath}: {e}")
        return None, None, None

# ------------------------------
# Main processing loop. Currently, we only work with sequences in .mp4 format. 
# This can be extended to other formats as well.
# ------------------------------
for mp4_path in Path(args.dist_dir).rglob("*.mp4"):
    rel_path = mp4_path.relative_to(args.dist_dir)
    split = rel_path.parts[0]
    name = mp4_path.stem
    base_name = name.split("_crf")[0]

    crf_match = re.search(r"_crf(\d+)", name)
    crf = crf_match.group(1) if crf_match else "unknown"

    # Match reference file
    ref_candidates = list(Path(args.ref_dir).joinpath(split).glob(f"{base_name}.*"))
    if not ref_candidates:
        print(f"Reference not found for: {base_name} in {split}")
        continue

    ref_path = ref_candidates[0]
    ref_ext = ref_path.suffix.lower().lstrip(".")

    if ref_ext not in args.input_formats:
        print(f"Unsupported reference format: {ref_path}")
        continue

    if ref_ext == "yuv":
        if not all([args.width, args.height, args.fps]):
            print(f"Skipping .yuv file {ref_path.name}: requires --width, --height, --fps")
            continue
        w, h, fps = str(args.width), str(args.height), str(args.fps)
    else:
        w, h, fps = get_metadata_with_ffprobe(ref_path)
        if not all([w, h, fps]):
            print(f"Could not extract metadata from reference file: {ref_path}")
            continue

    log_subdir = Path(args.log_dir) / split
    log_subdir.mkdir(parents=True, exist_ok=True)
    log_path = log_subdir / f"{base_name}_crf{crf}_vmaf.json"

    cmd = [ffmpeg_bin, "-y"]
    if ref_ext == "yuv":
        cmd += [
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "-s:v", f"{w}x{h}",
            "-r", fps
        ]

    cmd += [
        "-i", str(ref_path),
        "-i", str(mp4_path),
        "-lavfi", f"[0:v][1:v]libvmaf=log_path='{log_path.as_posix()}':log_fmt=json",
        "-f", "null", "-"
    ]

    print(f"Computing VMAF for {rel_path}")
    subprocess.run(cmd, check=True)

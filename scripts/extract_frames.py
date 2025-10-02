#!/usr/bin/env python3

import os
import re
import argparse
from pathlib import Path
import subprocess
import json

# ------------------------------
# CLI Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(description="Extract grayscale frames and group them for proxy VMAF training.")
parser.add_argument("--ref_dir", type=str, required=True, help="Directory with reference videos (mirrored layout)")
parser.add_argument("--dist_dir", type=str, required=True, help="Directory with compressed videos (mirrored layout)")
parser.add_argument("--frame_dir", type=str, required=True, help="Directory to store extracted frames")
parser.add_argument("--width", type=int, help="Width of .yuv reference (required if .yuv used)")
parser.add_argument("--height", type=int, help="Height of .yuv reference")
parser.add_argument("--fps", type=int, help="FPS of .yuv reference")
parser.add_argument("--input_formats", nargs="+", default=["yuv", "mp4", "mkv", "y4m"], help="Supported formats")
parser.add_argument("--ffmpeg_dir", type=str, default="", help="Directory containing ffmpeg and ffprobe binaries (optional)")
args = parser.parse_args()

# ------------------------------
# Setup
# ------------------------------
ffmpeg_bin = os.path.join(args.ffmpeg_dir, "ffmpeg") if args.ffmpeg_dir else "ffmpeg"
ffprobe_bin = os.path.join(args.ffmpeg_dir, "ffprobe") if args.ffmpeg_dir else "ffprobe"

orig_out = Path(args.frame_dir) / "Original"
comp_out = Path(args.frame_dir) / "Compressed"

# ------------------------------
# Metadata extractor (for non-YUV)
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
        fps = eval(info['r_frame_rate'])
        return str(width), str(height), str(int(fps))
    except Exception as e:
        print(f"Failed to get metadata from {filepath}: {e}")
        return None, None, None

# ------------------------------
# Frame Extraction
# ------------------------------
def extract_frames(input_path, output_dir, width, height, fps, is_yuv=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%04d.png"

    cmd = [ffmpeg_bin, "-y"]
    if is_yuv:
        cmd += [
            "-s", f"{width}x{height}",
            "-pix_fmt", "yuv420p",
            "-f", "rawvideo"
        ]
    cmd += [
        "-r", str(fps),
        "-i", str(input_path),
        "-vf", "format=gray",
        "-start_number", "0",
        str(output_pattern)
    ]
    subprocess.run(cmd, check=True)

# ------------------------------
# Process Reference Videos
# ------------------------------
for ref_path in Path(args.ref_dir).rglob("*.*"):
    ext = ref_path.suffix.lower().lstrip(".")
    if ext not in args.input_formats:
        continue
    name = ref_path.stem
    split = ref_path.relative_to(args.ref_dir).parts[0]

    if ext == "yuv":
        if not all([args.width, args.height, args.fps]):
            print(f"Skipping .yuv {ref_path.name} (need --width, --height, --fps)")
            continue
        w, h, fps = str(args.width), str(args.height), str(args.fps)
    else:
        w, h, fps = get_metadata_with_ffprobe(ref_path)
        if not all([w, h, fps]):
            continue

    out_path = orig_out / split / name
    extract_frames(ref_path, out_path, w, h, fps, is_yuv=(ext == "yuv"))

# ------------------------------
# Process Compressed Videos. We currently, only process *.mp4 
# format for enoded sequences. Other formats can also be added.
# ------------------------------
for dist_path in Path(args.dist_dir).rglob("*.mp4"):
    name = dist_path.stem
    base_name = name.split("_crf")[0]

    crf_match = re.search(r"_crf(\d+)", name)
    crf = crf_match.group(1) if crf_match else "unknown"

    split = dist_path.relative_to(args.dist_dir).parts[0]
    w, h, fps = get_metadata_with_ffprobe(dist_path)
    if not all([w, h, fps]):
        continue

    out_path = comp_out / split / base_name / f"crf{crf}"
    extract_frames(dist_path, out_path, w, h, fps)


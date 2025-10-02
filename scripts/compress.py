#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path
import json

# ------------------------------
# CLI Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser(description="Compress video files using FFmpeg with CRF values and folder mirroring.")
parser.add_argument("--input_dir", type=str, required=True, help="Base input directory containing Training, Validation, Testing subfolders")
parser.add_argument("--output_dir", type=str, required=True, help="Base output directory to store compressed files")
parser.add_argument("--crf", type=int, nargs="+", required=True, help="List of CRF values to use (e.g. --crf 28 32 36)")
parser.add_argument("--input_formats", nargs="+", default=["yuv", "mp4", "mkv", "y4m"], help="Accepted input formats (default: yuv mp4 mkv y4m)")
parser.add_argument("--width", type=int, help="Width of input (required for .yuv)")
parser.add_argument("--height", type=int, help="Height of input (required for .yuv)")
parser.add_argument("--fps", type=int, help="frame rate of input (required for .yuv)")
parser.add_argument("--ffmpeg_dir", type=str, default="", help="Directory containing ffmpeg binary (optional)")
args = parser.parse_args()

# ------------------------------
# Set ffmpeg binary path
# ------------------------------
ffmpeg_bin = os.path.join(args.ffmpeg_dir, "ffmpeg") if args.ffmpeg_dir else "ffmpeg"

# ------------------------------
# Encoder Map. It's x265 for now. Other encoders can be added on as-needed basis
# ------------------------------
encoders = {
    "x265": "libx265",
}

# ------------------------------
# Helper: Extract metadata via ffprobe
# ------------------------------
def get_metadata_with_ffprobe(filepath):
    try:
        cmd = [
            os.path.join(args.ffmpeg_dir, "ffprobe") if args.ffmpeg_dir else "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json", str(filepath)
        ]
        output = subprocess.check_output(cmd).decode()
        info = json.loads(output)['streams'][0]
        width = info['width']
        height = info['height']
        fps = eval(info['r_frame_rate'])  # e.g., "30/1"
        return str(width), str(height), str(int(fps))
    except Exception as e:
        print(f"Could not get metadata from {filepath}: {e}")
        return None, None, None

# ------------------------------
# Main Loop
# ------------------------------
splits = ["Training", "Validation", "Testing"]
for split in splits:
    input_split_dir = Path(args.input_dir) / split
    output_split_dir = Path(args.output_dir) / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    for file in input_split_dir.iterdir():
        ext = file.suffix.lower().lstrip('.')
        if ext not in args.input_formats:
            continue

        name = file.stem
        input_path = file

        # Get metadata
        if ext == "yuv":
            if not all([args.width, args.height, args.fps]):
                print(f"Skipping {file.name}: .yuv requires --width, --height, and --fps")
                continue
            w, h, fps = str(args.width), str(args.height), str(args.fps)
        else:
            w, h, fps = get_metadata_with_ffprobe(input_path)
            if not all([w, h, fps]):
                print(f"Skipping {file.name}: could not extract metadata")
                continue

        # Compress with each CRF
        for crf_val in args.crf:
            output_file = f"{name}_crf{crf_val}.mp4"
            output_path = output_split_dir / output_file

            cmd = [ffmpeg_bin, "-y"]
            if ext == "yuv":
                cmd += [
                    "-f", "rawvideo",
                    "-s:v", f"{w}x{h}",
                    "-r", fps
                ]

            cmd += [
                "-pix_fmt", "yuv420p",
                "-i", str(input_path),
                "-c:v", encoders[args.encoder],
                "-crf", str(crf_val),
                str(output_path)
            ]

            print(f"Compressing {file.name} (split={split}, CRF={crf_val}) → {output_file}")
            subprocess.run(cmd, check=True)

print("All compression complete.")

#!/usr/bin/env python3
import os
import re
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.io as io
from torchvision.io import ImageReadMode


def _as_posix(p: str) -> str:
    return str(p).replace("\\", "/")

class FilterDataset(Dataset):

    def __init__(self, csv_file, root_dir, crop_size=128, fixed_crop=False, 
                 n_frames=1, split=None):
        
        csv_file = str(csv_file)
        # --- CSV read: make fsspec/gcsfs use ADC when on Vertex ---
        if csv_file.startswith("gs://"):
            self.data = pd.read_csv(csv_file, low_memory=False, storage_options={"token": "cloud"})
        else:
            self.data = pd.read_csv(csv_file, low_memory=False)
                
        if split:
            self.data = self.data[self.data["split"] == split].reset_index(drop=True)
              
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.fixed_crop = fixed_crop
        self.n_frames = n_frames

        if n_frames % 2 == 0:
            raise ValueError(f"--n_frames must be odd (got {n_frames})")

        # Collect frame keys
        all_dist_keys = self._sorted_keys(self.data.columns, "dist_frame_")
        all_ref_keys  = self._sorted_keys(self.data.columns, "ref_frame_")

        if len(all_dist_keys) != len(all_ref_keys):
            raise ValueError("Mismatched number of distorted vs reference frame columns.")
        if len(all_dist_keys) < n_frames:
            raise ValueError(f"Need at least {n_frames} frame columns (got {len(all_dist_keys)}).")

        center = len(all_dist_keys) // 2
        half = n_frames // 2
        start = max(0, min(center - half, len(all_dist_keys) - n_frames))
        self.dist_keys = all_dist_keys[start:start+n_frames]
        self.ref_keys  = all_ref_keys[start:start+n_frames]

        self.skipped = 0

    def _sorted_keys(self, cols, prefix):
        """
        Return columns starting with `prefix` sorted by temporal index.

        Accepts:
        - f"{prefix}<digits>"               → index = <digits>   (e.g., dist_frame_0)
        - f"{prefix}t[+/-<digits>]"        → index = 0/±off     (e.g., dist_frame_t-1)
        Allows an underscore or end right after the index (e.g., "_path").
        """
        cols = [str(c) for c in cols if str(c).startswith(prefix)]

        # Fast-path: explicit t-triplet names present → use them in [-1,0,1] order
        t_triplet = [f"{prefix}t-1_path", f"{prefix}t_path", f"{prefix}t+1_path"]
        if all(c in cols for c in t_triplet):
            return t_triplet

        # Generic: accept digits OR t[±k], followed by "_" or end
        pat = re.compile(
            rf"^{re.escape(prefix)}(?:(?P<num>\d+)|t(?:(?P<sign>[+-])(?P<off>\d+))?)(?:_|$)"
        )

        pairs = []
        for c in cols:
            m = pat.search(c)
            if not m:
                continue
            if m.group("num") is not None:
                idx = int(m.group("num"))
            else:
                off = int(m.group("off") or 0)
                idx = -off if m.group("sign") == "-" else off  # t=0, t-1=-1, t+1=+1
            pairs.append((idx, c))

        if not pairs:
            raise ValueError(
                f"No usable columns for prefix '{prefix}'. Found: {sorted(cols)}. "
                f"Expected names like '{prefix}0(_...)' or '{prefix}t', '{prefix}t-1', '{prefix}t+1'."
            )

        pairs.sort(key=lambda kv: kv[0])
        return [c for _, c in pairs]

    def __len__(self):
        return len(self.data)

    def load_and_crop_torch(self, relative_path, i, j):
        
        # in FilterDataset.load_and_crop_torch(...)
        rel = _as_posix(str(relative_path))
        if rel.startswith("gs://"):
            # strip scheme and bucket so it becomes relative to root_dir
            rel = rel[5:].split("/", 1)[1]
        rel = rel.lstrip("/")
        full_path = os.path.join(self.root_dir, os.path.normpath(rel))

        if not os.path.exists(full_path):
            raise FileNotFoundError(full_path)
        try:
            img_uint8 = io.read_image(full_path, mode=ImageReadMode.GRAY)  # (1,H,W) uint8
        except RuntimeError as e:
            # torchvision raises RuntimeError(Errno 2) for missing files
            if "No such file or directory" in str(e):
                raise FileNotFoundError(full_path)
            raise

        img_float = img_uint8.float() / 255.0  # To [0,1]
        H, W = img_float.shape[-2:]
        i = 0 if H <= self.crop_size else min(i, H - self.crop_size)
        j = 0 if W <= self.crop_size else min(j, W - self.crop_size)
        return img_float[..., i:i+self.crop_size, j:j+self.crop_size].contiguous()  # (1, Hc, Wc)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        try:
            #Hardcoding for UGC 720p dataset
            H, W = 720,  1280

            # One crop for all frames
            if self.fixed_crop:
                i = max(0, (H - self.crop_size) // 2)
                j = max(0, (W - self.crop_size) // 2)
            else:
                i = 0 if H <= self.crop_size else random.randint(0, H - self.crop_size)
                j = 0 if W <= self.crop_size else random.randint(0, W - self.crop_size)

            # Distorted stack (full list, like ref)
            x_dist = torch.cat([self.load_and_crop_torch(row[k], i, j) for k in self.dist_keys], dim=0)
            # Reference stack
            x_ref = torch.cat([self.load_and_crop_torch(row[k], i, j) for k in self.ref_keys], dim=0)
            # vmaf
            y = torch.tensor(float(row["vmaf_score"]) / 100.0, dtype=torch.float32)
            # CRF for conditioning
            crf = torch.tensor([row["crf"] / 51.0], dtype=torch.float32)
            
            return x_ref, x_dist, y, crf

        except (FileNotFoundError, OSError, RuntimeError) as e:
            self.skipped += 1
            print(f"Error loading sample {idx} (paths: {row[self.dist_keys].tolist()}, {row[self.ref_keys].tolist()}): {str(e)}")
            if self.skipped % 100 == 0:
                print(f"Skipped {self.skipped} samples so far")
            if self.skipped > len(self.data) * 0.5:  # Example threshold
                raise RuntimeError(f"Too many samples skipped ({self.skipped}/{len(self.data)})")
            return None
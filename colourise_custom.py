#!/Users/romitchakraborty/venvs/deoldify/bin/python 
import sys, torch
from pathlib import Path

# 1) Parse the filename argument
import argparse

import functools
from torch.serialization import add_safe_globals, safe_globals

# Option A: permanently allowlist for this process
add_safe_globals([functools.partial])

p = argparse.ArgumentParser(
    description="DeOldify local colorizer (M2/MPS or CPU)"
)
p.add_argument(
    "filename",
    help="Name of the video in video/source/, e.g. clip_v4.mp4"
)
p.add_argument(
    "--render-factor", "-r",
    type=int, default=26,
    help="Color saturation/detail (21–35 is typical)"
)
p.add_argument(
    "--batch-size", "-b",
    type=int, default=16,
    help="Number of frames per batch (tune to your GPU RAM)"
)
p.add_argument(
    "--fp16",
    action="store_true",
    help="Enable mixed-precision inference (MPS/Metal only)"
)
args = p.parse_args()

# 2) Pick MPS if available, else CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# 3) Make sure Python sees the DeOldify code
repo_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(repo_dir))

# 4) Tell DeOldify which device to use
from deoldify import device as dldv_device
from deoldify.device_id import DeviceId
# DeviceId.GPU0 on MPS, else CPU
dldv_device.set(device=DeviceId.GPU0 if device.type == "mps" else DeviceId.CPU)

# 5) Load & tweak the colorizer
from deoldify.visualize import get_video_colorizer
colorizer = get_video_colorizer()
colorizer.batch_size = args.batch_size
colorizer.fp16       = args.fp16

# 6) Build input/output paths
in_path  = repo_dir / "video" / "source" / args.filename
if not in_path.exists():
    print(f"❌ Input file not found: {in_path}", file=sys.stderr)
    sys.exit(1)

# Colorize
out_path = colorizer.colorize_from_file_name(
    str(in_path),
    render_factor=args.render_factor,
    watermarked=False
)

print("✅ Saved colorized video to:", out_path)

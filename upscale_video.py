#!/usr/bin/env python3
import argparse, subprocess, os
import cv2
import numpy as np
import torch

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet  # only if you want x4plus variants
try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except Exception:
    HAS_GFPGAN = False

def pick_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")

def build_upsampler(model_name: str, device, tile: int, scale: int = 4):
    """
    Returns (upsampler, scale).
    Extend this dict if you want other models.
    """
    cfgs = {
        "realesr-general-x4v3": dict(
            model=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                  num_conv=32, upscale=4, act_type='prelu'),
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            scale=4
        ),
        "realesr-general-wdn-x4v3": dict(
            model=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                  num_conv=32, upscale=4, act_type='prelu'),
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesr-general-wdn-x4v3.pth",
            scale=4
        ),
        # Heavier alternatives (much slower on your machine based on your summary):
        "RealESRGAN_x4plus": dict(
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                          num_block=23, num_grow_ch=32, scale=4),
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            scale=4
        ),
        "RealESRNet_x4plus": dict(
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                          num_block=23, num_grow_ch=32, scale=4),
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
            scale=4
        ),
    }
    if model_name not in cfgs:
        raise ValueError(f"Unknown model: {model_name}")
    cfg = cfgs[model_name]
    model = cfg["model"].to(device)
    upsampler = RealESRGANer(
        scale=cfg["scale"],
        model_path=cfg["model_path"],
        model=model,
        device=device,
        tile=tile,        # adjust for VRAM; 512 is a good start on MPS
        tile_pad=10,
        pre_pad=0,
        half=False        # half not supported on MPS
    )
    return upsampler, cfg["scale"]

def build_face_enhancer(upsampler, scale: int):
    if not HAS_GFPGAN:
        raise RuntimeError("GFPGAN not installed; pip install gfpgan to enable --face")
    # v1.4 is the usual default; change path if you have a local file
    return GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        upscale=scale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="Input video path")
    p.add_argument("-o", "--output", required=True, help="Output video path")
    p.add_argument("-m", "--model", default="realesr-general-x4v3",
                   help="Model key (default: realesr-general-x4v3)")
    p.add_argument("--face", action="store_true", help="Enable face enhancement (GFPGAN)")
    p.add_argument("--tile", type=int, default=512, help="Tile size (higher=faster, more memory)")
    p.add_argument("--crf", type=int, default=17, help="x264 CRF for output quality")
    p.add_argument("--preset", default="slow", help="x264 preset (faster|fast|medium|slow|slower)")
    args = p.parse_args()

    device = pick_device()
    upsampler, scale = build_upsampler(args.model, device, args.tile, scale=4)

    face_enhancer = None
    if args.face:
        face_enhancer = build_face_enhancer(upsampler, scale)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = w * scale, h * scale

    # ffmpeg writer: raw bgr24 frames -> H.264; copy original audio if present
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", f"{fps:.06f}",
        "-i", "-",                  # stdin: video
        "-i", args.input,           # original to pull audio
        "-map", "0:v", "-map", "1:a?",
        "-shortest",
        "-c:v", "libx264",
        "-preset", args.preset, "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        args.output
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # RealESRGAN/GFPGAN expect BGR ndarray
            if face_enhancer is not None:
                _, _, out_bgr = face_enhancer.enhance(
                    frame_bgr, has_aligned=False,
                    only_center_face=False, paste_back=True
                )
            else:
                out_bgr, _ = upsampler.enhance(frame_bgr, outscale=scale)

            # stream out
            proc.stdin.write(out_bgr.astype(np.uint8).tobytes())

    proc.stdin.close()
    proc.wait()
    cap.release()
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()

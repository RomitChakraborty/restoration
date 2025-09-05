#!/Users/romitchakraborty/venvs/realesrgan311/bin/python 

import torch
import cv2
import os
import time
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def setup_device():
    """Setup the best available device for Apple Silicon"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (slower)")
    
    return device

def upscale_optimized(input_path, output_path, scale=4, tile_size=512):
    """
    Optimized upscaling for Apple Silicon M2 Pro
    
    Args:
        input_path: Path to input image
        output_path: Path to save upscaled image  
        scale: Upscaling factor
        tile_size: Tile size for processing (larger = faster but more memory)
    """
    
    device = setup_device()
    
    # Optimize for M2 Pro - you can use larger tiles with 16GB RAM
    if device.type == "mps":
        tile_size = 1024  # Larger tiles for M2 Pro
        half_precision = False  # MPS doesn't support half precision yet
    else:
        half_precision = True
    
    print(f"Using tile size: {tile_size}")
    
    try:
        # Setup model optimized for your hardware
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        #model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
         #              num_block=23, num_grow_ch=32, scale=4)
        
        # Move model to device
        model = model.to(device)
        
        # Initialize upsampler with optimizations
        upsampler = RealESRGANer(
            scale=4,
            model_path= 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesr-general-wdn-x4v3.pth',
            model=model,
            device=device,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=False,  # Use half precision if supported
            gpu_id=None  # Let it auto-detect
        )
        
        # Load image
        print(f"Loading: {input_path}")
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"❌ Error: Could not load {input_path}")
            return False
            
        print(f"Original size: {img.shape[1]}x{img.shape[0]}")
        
        # Time the upscaling
        start_time = time.time()
        print("🚀 Starting upscaling...")
        
        # Upscale with optimal settings
        with torch.inference_mode():  # Optimize inference
            output, _ = upsampler.enhance(img, outscale=scale)
        
        elapsed = time.time() - start_time
        print(f"⏱️  Upscaling completed in {elapsed:.2f} seconds")
        print(f"New size: {output.shape[1]}x{output.shape[0]}")
        
        # Save result
        cv2.imwrite(output_path, output)
        print(f"✅ Saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def batch_upscale(input_dir, output_dir, scale=4):
    """Batch upscale all images in a directory"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for ext in extensions:
        image_files.extend([f for f in os.listdir(input_dir) 
                           if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} images to process")
    
    total_start = time.time()
    
    for i, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n[{i+1}/{len(image_files)}] Processing: {filename}")
        
        success = upscale_optimized(input_path, output_path, scale)
        
        if not success:
            print(f"❌ Failed to process {filename}")
    
    total_elapsed = time.time() - total_start
    print(f"\n🏁 Batch processing completed in {total_elapsed:.2f} seconds")
    print(f"Average: {total_elapsed/len(image_files):.2f} seconds per image")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Real-ESRGAN for Apple Silicon')
    parser.add_argument('-i', '--input', required=True, help='Input image or directory')
    parser.add_argument('-o', '--output', required=True, help='Output image or directory')
    parser.add_argument('-s', '--scale', type=int, default=4, help='Upscaling factor')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--tile', type=int, default=512, help='Tile size (larger=faster but more memory)')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_upscale(args.input, args.output, args.scale)
    else:
        upscale_optimized(args.input, args.output, args.scale, args.tile)

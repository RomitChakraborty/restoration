#!/Users/romitchakraborty/venvs/realesrgan311/bin/python 

import torch
import cv2
import os
import time
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer

def setup_device():
    """Setup the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def get_model_config(model_name):
    """Get model configuration for different Real-ESRGAN variants"""
    configs = {
        'RealESRNet_x4plus': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
            'scale': 4
        },
        'RealESRGAN_x4plus': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'scale': 4
        },
         'realesr-general-x4v3': {
            'model': SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
            'scale': 4
        },
        'realesr-general-wdn-x4v3': {
            'model': SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesr-general-wdn-x4v3.pth',
            'scale': 4
        }
    }
    return configs.get(model_name)

def upscale_with_face_enhance(input_path, output_path, model_name, device, face_enhance=True, tile_size=1024):
    """Upscale image with specified model and optional face enhancement"""
    
    config = get_model_config(model_name)
    if not config:
        print(f"Unknown model: {model_name}")
        return False, 0
    
    try:
        suffix = "_face_enhanced" if face_enhance else "_no_face_enhance"
        actual_output = output_path.replace('.png', f'{suffix}.png')
        
        print(f"\n--- Testing {model_name} {'WITH' if face_enhance else 'WITHOUT'} face enhancement ---")
        
        # Setup model
        model = config['model'].to(device)
        
        # Initialize upsampler
        upsampler = RealESRGANer(
            scale=config['scale'],
            model_path=config['model_path'],
            model=model,
            device=device,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=False  # MPS doesn't support half precision
        )
        
        # Setup face enhancer if needed
        face_enhancer = None
        if face_enhance:
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=1,  # Don't upscale faces separately, just enhance
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler
            )
        
        # Load image
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            return False, 0
        
        print(f"Processing {'with face enhancement' if face_enhance else 'without face enhancement'}...")
        start_time = time.time()
        
        # Process image
        with torch.inference_mode():
            if face_enhance and face_enhancer:
                # Use GFPGAN with background upscaling
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                # Just upscale
                output, _ = upsampler.enhance(img, outscale=config['scale'])
        
        elapsed = time.time() - start_time
        
        # Save result
        cv2.imwrite(actual_output, output)
        
        print(f"Completed in {elapsed:.1f} seconds")
        print(f"Saved: {actual_output}")
        
        return True, elapsed
        
    except Exception as e:
        print(f"Error with {model_name} (face_enhance={face_enhance}): {e}")
        return False, 0

def compare_face_enhancement(input_path, output_dir="face_enhance_comparison"):
    """Compare models with and without face enhancement"""
    
    device = setup_device()
    
    # Test the two best performing models from your previous test
    models_to_test = [
        'realesr-general-wdn-x4v3',  # Fastest
        'realesr-general-x4v3',
        'RealESRGAN_x4plus',
        'RealESRNet_x4plus',         # Good quality/speed balance
    ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    print(f"Comparing face enhancement on: {input_path}")
    print("=" * 70)
    
    for model_name in models_to_test:
        # Test without face enhancement
        output_path = os.path.join(output_dir, f"{model_name}_result.png")
        success, elapsed = upscale_with_face_enhance(
            input_path, output_path, model_name, device, face_enhance=False
        )
        
        if success:
            results[f"{model_name}_no_face"] = {
                'time': elapsed,
                'output': output_path.replace('.png', '_no_face_enhance.png')
            }
        
        # Test with face enhancement
        success, elapsed = upscale_with_face_enhance(
            input_path, output_path, model_name, device, face_enhance=True
        )
        
        if success:
            results[f"{model_name}_face"] = {
                'time': elapsed,
                'output': output_path.replace('.png', '_face_enhanced.png')
            }
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY:")
    print("=" * 70)
    
    for model_name, data in results.items():
        print(f"{model_name:<30} | {data['time']:>6.1f}s | {data['output']}")
    
    print("\n" + "=" * 70)
    print("WHAT TO LOOK FOR:")
    print("- Face detail improvement (skin texture, eyes, hair)")
    print("- Overall image sharpness vs naturalness") 
    print("- Processing time vs quality trade-off")
    print("- Potential over-smoothing or artifacts on faces")
    print("=" * 70)
    
    print(f"\nAll results saved in: {output_dir}/")
    print("Open the directory and compare images at 100% zoom to see face detail differences.")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Real-ESRGAN with/without face enhancement')
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('-o', '--output-dir', default='face_enhance_comparison', help='Output directory')
    
    args = parser.parse_args()
    
    compare_face_enhancement(args.input, args.output_dir)
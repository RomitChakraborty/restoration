#!/Users/romitchakraborty/venvs/realesrgan311/bin/python 

import torch
import cv2
import os
import time
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

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
        'RealESRGAN_x4plus': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'scale': 4
        },
        'RealESRNet_x4plus': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'model_path': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
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

def upscale_with_model(input_path, output_path, model_name, device, tile_size=1024):
    """Upscale image with specified model"""
    
    config = get_model_config(model_name)
    if not config:
        print(f"Unknown model: {model_name}")
        return False, 0
    
    try:
        print(f"\n--- Testing {model_name} ---")
        
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
        
        # Load image
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            return False, 0
        
        print(f"Processing with {model_name}...")
        start_time = time.time()
        
        # Upscale
        with torch.inference_mode():
            output, _ = upsampler.enhance(img, outscale=config['scale'])
        
        elapsed = time.time() - start_time
        
        # Save result
        cv2.imwrite(output_path, output)
        
        print(f"Completed in {elapsed:.1f} seconds")
        print(f"Saved: {output_path}")
        
        return True, elapsed
        
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        return False, 0

def compare_models(input_path, output_dir="model_comparisons"):
    """Compare multiple models on the same image"""
    
    device = setup_device()
    
    # Models to test (ordered from most conservative to most aggressive)
    models_to_test = [
        'RealESRNet_x4plus',        # Most conservative
        'RealESRGAN_x4plus',        # Standard
        'realesr-general-x4v3',     # Better for real-world degradation
        'realesr-general-wdn-x4v3'  # Best for noisy/degraded content
    ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    print(f"Comparing {len(models_to_test)} models on: {input_path}")
    print("=" * 60)
    
    for model_name in models_to_test:
        output_path = os.path.join(output_dir, f"{model_name}_result.png")
        
        success, elapsed = upscale_with_model(
            input_path, output_path, model_name, device
        )
        
        if success:
            results[model_name] = {
                'time': elapsed,
                'output': output_path
            }
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY:")
    print("=" * 60)
    
    for model_name, data in results.items():
        print(f"{model_name:<25} | {data['time']:>6.1f}s | {data['output']}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR VINTAGE FOOTAGE:")
    print("- RealESRNet_x4plus: Most natural, preserves original texture")
    print("- realesr-general-wdn-x4v3: Best for noisy/compressed vintage content") 
    print("- realesr-general-x4v3: Good balance for real-world degradation")
    print("- RealESRGAN_x4plus: Most aggressive enhancement")
    print("=" * 60)
    
    print(f"\nAll results saved in: {output_dir}/")
    print("Compare the images visually to choose the best model for your video.")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Real-ESRGAN models')
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('-o', '--output-dir', default='model_comparisons', help='Output directory')
    
    args = parser.parse_args()
    
    compare_models(args.input, args.output_dir)

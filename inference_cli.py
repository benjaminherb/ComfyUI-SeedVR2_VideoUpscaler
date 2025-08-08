#!/usr/bin/env python3
"""
Standalone SeedVR2 Video Upscaler CLI Script
"""

import sys
import os
import argparse
import time
import multiprocessing as mp

# Set up path before any other imports to fix module resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Set environment variable so all spawned processes can find modules
os.environ['PYTHONPATH'] = script_dir + ':' + os.environ.get('PYTHONPATH', '')

# Ensure safe CUDA usage with multiprocessing
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)
# -------------------------------------------------------------
# 1) Gestion VRAM (cudaMallocAsync) déjà en place
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

# 2) Pré-parse de la ligne de commande pour récupérer --cuda_device
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--cuda_device", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.cuda_device is not None:
    device_list_env = [x.strip() for x in _pre_args.cuda_device.split(',') if x.strip()!='']
    if len(device_list_env) == 1:
        # Single GPU: restrict visibility now
        os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]

# -------------------------------------------------------------
# 3) Imports lourds (torch, etc.) après la configuration env
import torch
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from src.utils.downloads import download_weight
from src.utils.debug import Debug

debug = Debug(enabled=False)  # Default to disabled, can be enabled via CLI


def extract_frames_from_video(video_path, skip_first_frames=0, load_cap=None):
    """
    Extract frames from video and convert to tensor format
    
    Args:
        video_path (str): Path to input video
        skip_first_frame (bool): Skip the first frame during extraction
        load_cap (int): Maximum number of frames to load (None for all)
        
    Returns:
        torch.Tensor: Frames tensor in format [T, H, W, C] (Float16, normalized 0-1)
    """
    debug.log(f"Extracting frames from video: {video_path}", category="file")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    debug.log(f"Video info: {frame_count} frames, {width}x{height}, {fps:.2f} FPS", category="info")
    if skip_first_frames:
        debug.log(f"Will skip first {skip_first_frames} frames", category="info")
    if load_cap:
        debug.log(f"Will load maximum {load_cap} frames", category="info")
    
    frames = []
    frame_idx = 0
    frames_loaded = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip first frame if requested
        if frame_idx < skip_first_frames:
            frame_idx += 1
            debug.log(f"Skipped first frame", category="info")
            continue
        
        # Check load cap
        if load_cap is not None and load_cap > 0 and frames_loaded >= load_cap:
            debug.log(f"Reached load cap of {load_cap} frames", category="info")
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to 0-1
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
        frame_idx += 1
        frames_loaded += 1
        
        if debug.enabled and frames_loaded % 100 == 0:
            total_to_load = min(frame_count, load_cap) if load_cap else frame_count
            debug.log(f"Extracted {frames_loaded}/{total_to_load} frames", category="file")
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    debug.log(f"Extracted {len(frames)} frames", category="success")
    
    # Convert to tensor [T, H, W, C] and cast to Float16 for ComfyUI compatibility
    frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.float16)
    
    debug.log(f"Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}", category="memory")

    return frames_tensor, fps


def save_frames_to_video(frames_tensor, output_path, fps=30.0):
    """
    Save frames tensor to video file
    
    Args:
        frames_tensor (torch.Tensor): Frames in format [T, H, W, C] (Float16, 0-1)
        output_path (str): Output video path
        fps (float): Output video FPS
    """
    debug.log(f"Saving {frames_tensor.shape[0]} frames to video: {output_path}", category="file")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert tensor to numpy and denormalize
    frames_np = frames_tensor.cpu().numpy()
    frames_np = (frames_np * 255.0).astype(np.uint8)
    
    # Get video properties
    T, H, W, C = frames_np.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create video writer for: {output_path}")
    
    # Write frames
    for i, frame in enumerate(frames_np):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        if debug.enabled and (i + 1) % 100 == 0:
            debug.log(f"Saved {i + 1}/{T} frames", category="file")

    out.release()
    
    debug.log(f"Video saved successfully: {output_path}", category="success")


def save_frames_to_png(frames_tensor, output_dir, base_name):
    """
    Save frames tensor as sequential PNG images.

    Args:
        frames_tensor (torch.Tensor): Frames in format [T, H, W, C] (Float16, 0-1)
        output_dir (str): Directory to save PNGs
        base_name (str): Base name for output files (without extension)
    """
    debug.log(f"Saving {frames_tensor.shape[0]} frames as PNGs to directory: {output_dir}", category="file")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy uint8 RGB
    frames_np = (frames_tensor.cpu().numpy() * 255.0).astype(np.uint8)
    total = frames_np.shape[0]
    digits = max(5, len(str(total)))  # at least 5 digits

    for idx, frame in enumerate(frames_np):
        filename = f"{base_name}_{idx:0{digits}d}.png"
        file_path = os.path.join(output_dir, filename)
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, frame_bgr)
        if debug.enabled and (idx + 1) % 100 == 0:
            debug.log(f"Saved {idx + 1}/{total} PNGs", category="file")

    debug.log(f"PNG saving completed: {total} files in '{output_dir}'", category="success")


def _worker_process(proc_idx, device_id, frames_np, shared_args, return_queue):
    """Worker process that performs upscaling on a slice of frames using a dedicated GPU."""
    # 1. Limit CUDA visibility to the chosen GPU BEFORE importing torch-heavy deps
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # Keep same cudaMallocAsync setting
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    import torch  # local import inside subprocess
    from src.core.model_manager import configure_runner
    from src.core.generation import generation_loop
    
    # Create debug instance for this worker process
    worker_debug = Debug(enabled=shared_args["debug"])
    
    # Reconstruct frames tensor
    frames_tensor = torch.from_numpy(frames_np).to(torch.float16)

    # Prepare runner
    model_dir = shared_args["model_dir"]
    model_name = shared_args["model"]
    # ensure model weights present (each process checks but very fast if already downloaded)
    worker_debug.log(f"Configuring runner for device {device_id}", category="general")
    runner = configure_runner(model_name, model_dir, shared_args["preserve_vram"], worker_debug, block_swap_config=shared_args["block_swap_config"], vae_tiling_enabled=shared_args["vae_tiling_enabled"], vae_tile_size=shared_args["vae_tile_size"], vae_tile_overlap=shared_args["vae_tile_overlap"])

    # Run generation
    result_tensor = generation_loop(
        runner=runner,
        images=frames_tensor,
        cfg_scale=shared_args["cfg_scale"],
        seed=shared_args["seed"],
        res_w=shared_args["res_w"],
        batch_size=shared_args["batch_size"],
        preserve_vram=shared_args["preserve_vram"],
        temporal_overlap=shared_args["temporal_overlap"],
        debug=worker_debug,
        block_swap_config=shared_args["block_swap_config"]

    )

    # Send back result as numpy array to avoid CUDA transfers
    return_queue.put((proc_idx, result_tensor.cpu().numpy()))


def _gpu_processing(frames_tensor, device_list, args):
    """Split frames and process them in parallel on multiple GPUs."""
    num_devices = len(device_list)
    # split frames tensor along time dimension
    chunks = torch.chunk(frames_tensor, num_devices, dim=0)

    manager = mp.Manager()
    return_queue = manager.Queue()
    workers = []

    shared_args = {
        "model": args.model,
        "model_dir": args.model_dir if args.model_dir is not None else "./models/SEEDVR2",
        "preserve_vram": args.preserve_vram,
        "debug": args.debug,
        "cfg_scale": 1.0,
        "seed": args.seed,
        "res_w": args.resolution,
        "batch_size": args.batch_size,
        "temporal_overlap": 0,
        "block_swap_config": {
            'blocks_to_swap': args.blocks_to_swap,
            'use_none_blocking': args.use_none_blocking,
            'offload_io_components': args.offload_io_components,
            'cache_model': args.cache_model,
        },
        "vae_tiling_enabled": args.vae_tiling_enabled,
        "vae_tile_size": args.vae_tile_size,
        "vae_tile_overlap": args.vae_tile_overlap,
    }

    for idx, (device_id, chunk_tensor) in enumerate(zip(device_list, chunks)):
        p = mp.Process(
            target=_worker_process,
            args=(idx, device_id, chunk_tensor.cpu().numpy(), shared_args, return_queue),
        )
        p.start()
        workers.append(p)

    results_np = [None] * num_devices
    collected = 0
    while collected < num_devices:
        proc_idx, res_np = return_queue.get()
        results_np[proc_idx] = res_np
        collected += 1

    for p in workers:
        p.join()

    # Concatenate results in original order
    result_tensor = torch.from_numpy(np.concatenate(results_np, axis=0)).to(torch.float16)
    return result_tensor


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SeedVR2 Video Upscaler CLI")
    
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed for generation (default: 100)")
    parser.add_argument("--resolution", type=int, default=1072,
                        help="Target resolution of the short side (default: 1072)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of frames per batch (default: 5)")
    parser.add_argument("--model", type=str, default="seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                        choices=[
                            "seedvr2_ema_3b_fp16.safetensors",
                            "seedvr2_ema_3b_fp8_e4m3fn.safetensors", 
                            "seedvr2_ema_7b_fp16.safetensors",
                            "seedvr2_ema_7b_fp8_e4m3fn.safetensors"
                        ],
                        help="Model to use (default: 3B FP8)")
    parser.add_argument("--model_dir", type=str, default="seedvr2_models",
                            help="Directory containing the model files (default: use cache directory)")
    parser.add_argument("--skip_first_frames", type=int, default=0,
                        help="Skip the first frames during processing")
    parser.add_argument("--load_cap", type=int, default=0,
                        help="Maximum number of frames to load from video (default: load all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated, if output_format is png, it will be a directory)")
    parser.add_argument("--output_format", type=str, default="video", choices=["video", "png"],
                        help="Output format: 'video' (mp4) or 'png' images (default: video)")
    parser.add_argument("--preserve_vram", action="store_true",
                        help="Enable VRAM preservation mode")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--cuda_device", type=str, default=None,
                        help="CUDA device id(s). Single id (e.g., '0') or comma-separated list '0,1' for multi-GPU")
    parser.add_argument("--blocks_to_swap", type=int, default=0,
                        help="Number of blocks to swap for VRAM optimization (default: 0, disabled), up to 32 for 3B model, 36 for 7B")
    parser.add_argument("--use_none_blocking", action="store_true",
                        help="Use non-blocking memory transfers for VRAM optimization")
    parser.add_argument("--cache_model", action="store_true",
                        help="Cache model weights in memory to avoid reloading")
    parser.add_argument("--offload_io_components", action="store_true",
                        help="Offload IO components to CPU for VRAM optimization")
    parser.add_argument("--vae_tiling_enabled", action="store_true",
                        help="Enable VAE tiling for improved VRAM usage")
    parser.add_argument("--vae_tile_size", type=int, default=512,
                        help="VAE tile size for tiled decoding (default: 512). Only used if --vae_tiling_enabled is set")
    parser.add_argument("--vae_tile_overlap", type=int, default=128,
                        help="VAE tile overlap for tiled decoding (default: 128). Only used if --vae_tiling_enabled is set")

    return parser.parse_args()


def main():
    """Main CLI function"""
    debug.log(f"SeedVR2 Video Upscaler CLI started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", category="model", force=True)
    
    # Parse arguments
    args = parse_arguments()
    debug.enabled = args.debug  
    
    debug.log("Arguments:", category="setup")
    for key, value in vars(args).items():
        debug.log(f"  {key}: {value}", category="none")
    
    # Show actual CUDA device visibility
    debug.log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all)')}", category="device")
    if torch.cuda.is_available():
        debug.log(f"torch.cuda.device_count(): {torch.cuda.device_count()}", category="device")
        debug.log(f"Using device index 0 inside script (mapped to selected GPU)", category="device")
    
    try:
        # Ensure --output is a directory when using PNG format
        if args.output_format == "png":
            output_path_obj = Path(args.output)
            if output_path_obj.suffix:  # an extension is present, strip it
                args.output = str(output_path_obj.with_suffix(''))
        
        debug.log(f"Output will be saved to: {args.output}", category="file")
        
        # Extract frames from video
        debug.log(f"Extracting frames from video...", category="generation")
        start_time = time.time()
        frames_tensor, original_fps = extract_frames_from_video(
            args.video_path, 
            args.skip_first_frames, 
            args.load_cap
        )
        
        debug.log(f"Frame extraction time: {time.time() - start_time:.2f}s", category="general")
        # debug.log(f"📊 Initial VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB", category="memory")

        # Parse GPU list
        device_list = [d.strip() for d in str(args.cuda_device).split(',') if d.strip()] if args.cuda_device else ["0"]
        debug.log(f"Using devices: {device_list}", category="device")
        processing_start = time.time()
        download_weight(args.model, args.model_dir)
        result = _gpu_processing(frames_tensor, device_list, args)
        generation_time = time.time() - processing_start
        
        debug.log(f"Generation time: {generation_time:.2f}s", category="general")
        debug.log(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB", category="memory")
        debug.log(f"Result shape: {result.shape}, dtype: {result.dtype}", category="memory")
        
        # After generation_time calculation, choose saving method
        if args.output_format == "png":
            # Ensure output treated as directory
            output_dir = args.output
            base_name = Path(args.video_path).stem + "_upscaled"
            debug.log(f"Saving PNG frames to directory: {output_dir}", category="file")
            
            save_start = time.time()
            save_frames_to_png(result, output_dir, base_name)

            debug.log(f"Save time: {time.time() - save_start:.2f}s", category="general")

        else:
            # Save video
            debug.log(f"Saving upscaled video to: {args.output}", category="file")
            save_start = time.time()
            save_frames_to_video(result, args.output, original_fps)
            debug.log(f"Save time: {time.time() - save_start:.2f}s", category="general")
        
        total_time = time.time() - start_time
        debug.log(f"Upscaling completed successfully!", category="success", force=True)
        if args.output_format == "png":
            debug.log(f"PNG frames saved in directory: {args.output}", category="file", force=True)
        else:
            debug.log(f"Output saved to video: {args.output}", category="file", force=True)
        debug.log(f"Total processing time: {total_time:.2f}s", category="timing", force=True)
        debug.log(f"Average FPS: {len(frames_tensor) / generation_time:.2f} frames/sec", category="timing", force=True)
        
    except Exception as e:
        debug.log(f"Error during processing: {e}", category="error", force=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        debug.log(f"Process {os.getpid()} terminating - VRAM will be automatically freed", category="cleanup", force=True)


if __name__ == "__main__":
    main() 
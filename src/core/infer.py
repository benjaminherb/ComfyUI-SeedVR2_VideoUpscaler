# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import time
from typing import List, Optional, Tuple, Union
import torch
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from src.optimization.memory_manager import clear_vram_cache

from src.common.diffusion import (
    classifier_free_guidance_dispatcher,
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
)
from src.common.distributed import (
    get_device,
)

# from common.fs import download

from src.models.dit_v2 import na


def optimized_channels_to_last(tensor):
    """🚀 Optimized replacement for rearrange(tensor, 'b c ... -> b ... c')
    Moves channels from position 1 to last position using PyTorch native operations.
    """
    if tensor.ndim == 3:  # [batch, channels, spatial]
        return tensor.permute(0, 2, 1)
    elif tensor.ndim == 4:  # [batch, channels, height, width]
        return tensor.permute(0, 2, 3, 1)
    elif tensor.ndim == 5:  # [batch, channels, depth, height, width]
        return tensor.permute(0, 2, 3, 4, 1)
    else:
        # Fallback for other dimensions - move channel (dim=1) to last
        dims = list(range(tensor.ndim))
        dims = [dims[0]] + dims[2:] + [dims[1]]  # [0, 2, 3, ..., 1]
        return tensor.permute(*dims)

def optimized_channels_to_second(tensor):
    """🚀 Optimized replacement for rearrange(tensor, 'b ... c -> b c ...')
    Moves channels from last position to position 1 using PyTorch native operations.
    """
    if tensor.ndim == 3:  # [batch, spatial, channels]
        return tensor.permute(0, 2, 1)
    elif tensor.ndim == 4:  # [batch, height, width, channels]
        return tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 5:  # [batch, depth, height, width, channels]
        return tensor.permute(0, 4, 1, 2, 3)
    else:
        # Fallback for other dimensions - move last dim to position 1
        dims = list(range(tensor.ndim))
        dims = [dims[0], dims[-1]] + dims[1:-1]  # [0, -1, 1, 2, ..., -2]
        return tensor.permute(*dims)

class VideoDiffusionInfer():
    def __init__(self, config: DictConfig, debug=None,  vae_tiling_enabled: bool = False, 
                 vae_tile_size: int = 512, vae_tile_overlap: int = 64):
        # Check if debug instance is available
        if debug is None:
            raise ValueError("Debug instance must be provided to VideoDiffusionInfer")
        self.config = config
        self.debug = debug
        self.vae_tiling_enabled = vae_tiling_enabled
        self.vae_tile_size = vae_tile_size
        self.vae_tile_overlap = vae_tile_overlap
        
    def get_condition(self, latent: Tensor, latent_blur: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            if task == "sr":
                cond[:, ..., :-1] = latent_blur[:]
                cond[:, ..., -1:] = 1.0
            return cond
        if task == "i2v":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        if task == "sr":
            # sr generation.
            cond[:, ..., :-1] = latent_blur[:]
            cond[:, ..., -1:] = 1.0
            return cond
        raise NotImplementedError
    
    def configure_diffusion(self):
        self.schedule = create_schedule_from_config(
            config=self.config.diffusion.schedule,
            device=get_device(),
        )
        self.sampling_timesteps = create_sampling_timesteps_from_config(
            config=self.config.diffusion.timesteps.sampling,
            schedule=self.schedule,
            device=get_device(),
        )
        self.sampler = create_sampler_from_config(
            config=self.config.diffusion.sampler,
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
        )

    # -------------------------------- Helper ------------------------------- #

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor], preserve_vram: bool = False) -> List[Tensor]:
        use_sample = self.config.vae.get("use_sample", True)
        latents = []
        if len(samples) > 0:
            device = get_device()
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group samples of the same shape to batches if enabled.
            if self.config.vae.grouping:
                batches, indices = na.pack(samples)
            else:
                batches = [sample.unsqueeze(0) for sample in samples]

            # VAE process by each group.
            for sample in batches:
                sample = sample.to(device, dtype)
                if hasattr(self.vae, "preprocess"):
                    sample = self.vae.preprocess(sample)

                # Decide on tiling (use output-space size)
                H = sample.shape[-2] if sample.ndim >= 4 else 0
                W = sample.shape[-1] if sample.ndim >= 4 else 0
                spatial_size = H * W
                use_tiling = (hasattr(self, 'vae_tiling_enabled') and self.vae_tiling_enabled and
                    spatial_size > 256 * 256)  # threshold on output resolution

                out = self.vae.encode(
                    sample,
                    preserve_vram=preserve_vram,
                    tiled=use_tiling,
                    tile_size=self.vae_tile_size,
                    tile_overlap=self.vae_tile_overlap,
                )

                posterior = getattr(out, "latent_dist", getattr(out, "posterior", out))
                if use_sample:
                    latent = posterior.rsample() if hasattr(posterior, "rsample") else posterior.sample()
                else:
                    latent = posterior.mode()

                # Ensure 5D then move channels last and apply scale/shift
                if latent.ndim == 4:
                    latent = latent.unsqueeze(2)
                latent = rearrange(latent, "b c ... -> b ... c")
                latent = (latent - shift) * scale
                latents.append(latent)

                clear_vram_cache(self.debug)

            # Ungroup back to individual latent with the original order.
            if self.config.vae.grouping:
                latents = na.unpack(latents, indices)
            else:
                latents = [latent.squeeze(0) for latent in latents]

        return latents
    

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor], target_dtype: torch.dtype = None, preserve_vram: bool = False) -> List[Tensor]:
        """🚀 VAE decode optimisé - décodage direct sans chunking, compatible avec autocast externe"""
        samples = []
        if len(latents) > 0:
            #t = time.time()
            device = get_device()
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Check if tiling is enabled and if the latents are large enough to warrant it
            # This is a heuristic, adjust the threshold if needed. 512*512 is a good starting point.
            first_latent = latents[0]
            spatial_size = first_latent.shape[1] * first_latent.shape[2] # H * W of latent
            use_tiling = (
                hasattr(self, 'vae_tiling_enabled') and self.vae_tiling_enabled and
                spatial_size > 32*32 # A threshold for latent size (e.g., > 256x256 image)
            )

            if use_tiling:
                self.debug.log(f"Using VAE Tiled Decoding (Tile: {self.vae_tile_size}, Overlap: {self.vae_tile_overlap})", category="vae", force=True)

                # Apply same grouping logic as regular decode
                if self.config.vae.grouping:
                    latents, indices = na.pack(latents)
                else:
                    latents = [latent.unsqueeze(0) for latent in latents]

                # Tiling is done one latent at a time
                for latent in latents:
                    effective_dtype = target_dtype if target_dtype is not None else dtype
                    latent = latent.to(device, effective_dtype, non_blocking=True)
                    latent = latent / scale + shift
                    latent = rearrange(latent, "b ... c -> b c ...")
                    
                    with torch.autocast("cuda", torch.float16, enabled=True):
                        sample = self.vae.tiled_decode(
                            latent,
                            tile_size=self.vae_tile_size,
                            tile_overlap=self.vae_tile_overlap,
                            preserve_vram=preserve_vram,
                        )
                    
                    if hasattr(self.vae, "postprocess"):
                        sample = self.vae.postprocess(sample)
                    samples.append(sample) 
                    clear_vram_cache(self.debug)

                if self.config.vae.grouping:
                    samples = na.unpack(samples, indices)
                else:
                    samples = [sample.squeeze(0) for sample in samples]

            else: # Original logic for smaller images or when tiling is disabled
                if self.config.vae.grouping:
                    latents, indices = na.pack(latents)
                else:
                    latents = [latent.unsqueeze(0) for latent in latents]
                t = time.time()
                for i, latent in enumerate(latents):
                    effective_dtype = target_dtype if target_dtype is not None else dtype
                    latent = latent.to(device, effective_dtype, non_blocking=True)
                    latent = latent / scale + shift
                    latent = rearrange(latent, "b ... c -> b c ...")
                    # Original logic used squeeze(2), let's keep it for compatibility
                    if latent.ndim == 5:
                        latent = latent.squeeze(2)
                    with torch.autocast("cuda", torch.float16, enabled=True):
                            sample = self.vae.decode(latent, preserve_vram=True).sample
                    if hasattr(self.vae, "postprocess"):
                        sample = self.vae.postprocess(sample)
                    samples.append(sample)
                    if i % 2 == 0 or i == len(latents) - 1:
                        torch.cuda.empty_cache()
                print(f"🔄 DECODE time: {time.time() - t} seconds")
                if self.config.vae.grouping:
                    samples = na.unpack(samples, indices)
                else:
                    samples = [sample.squeeze(0) for sample in samples]

        return samples


    def timestep_transform(self, timesteps: Tensor, latents_shapes: Tensor):
        # Skip if not needed.
        if not self.config.diffusion.timesteps.get("transform", False):
            return timesteps

        # Compute resolution.
        vt = self.config.vae.model.get("temporal_downsample_factor", 4)
        vs = self.config.vae.model.get("spatial_downsample_factor", 8)
        frames = (latents_shapes[:, 0] - 1) * vt + 1
        heights = latents_shapes[:, 1] * vs
        widths = latents_shapes[:, 2] * vs

        # Compute shift factor.
        def get_lin_function(x1, y1, x2, y2):
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b

        img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
        vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
        shift = torch.where(
            frames > 1,
            vid_shift_fn(heights * widths * frames),
            img_shift_fn(heights * widths),
        )

        # Shift timesteps.
        timesteps = timesteps / self.schedule.T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * self.schedule.T
        return timesteps

    def get_vram_usage(self):
        """Obtenir l'utilisation VRAM actuelle (allouée et réservée)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            return allocated, reserved, max_allocated
        return 0, 0, 0

    def get_vram_peak(self):
        """Obtenir le pic VRAM depuis le dernier reset"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
        return 0

    def reset_vram_peak(self):
        """Reset le compteur de pic VRAM"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def inference(
        self,
        noises: List[Tensor],
        conditions: List[Tensor],
        texts_pos: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        texts_neg: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        cfg_scale: Optional[float] = None,
        preserve_vram: bool = False,
        temporal_overlap: int = 0,
        use_blockswap: bool = False,
    ) -> List[Tensor]:
        assert len(noises) == len(conditions) == len(texts_pos) == len(texts_neg)
        batch_size = len(noises)

        # Return if empty.
        if batch_size == 0:
            return []

        # Monitoring VRAM initial et reset des pics
        #self.reset_vram_peak()
        
        # Set cfg scale
        if cfg_scale is None:
            cfg_scale = self.config.diffusion.cfg.scale

        # 🚀 OPTIMISATION: Use BFloat16 autocast for all models
        # - FP8 models: BFloat16 required for arithmetic operations
        # - FP16 models: BFloat16 provides better numerical stability and prevents black frames
        # - BFloat16 models: Already optimal
        target_dtype = torch.bfloat16

        model_dtype = next(self.dit.parameters()).dtype
        self.debug.log(f"Model dtype: {model_dtype}, using {target_dtype} for autocast", category="precision")
        
        # Text embeddings.
        assert type(texts_pos[0]) is type(texts_neg[0])
        if isinstance(texts_pos[0], str):
            text_pos_embeds, text_pos_shapes = self.text_encode(texts_pos)
            text_neg_embeds, text_neg_shapes = self.text_encode(texts_neg)
        elif isinstance(texts_pos[0], tuple):
            text_pos_embeds, text_pos_shapes = [], []
            text_neg_embeds, text_neg_shapes = [], []
            for pos in zip(*texts_pos):
                emb, shape = na.flatten(pos)
                text_pos_embeds.append(emb)
                text_pos_shapes.append(shape)
            for neg in zip(*texts_neg):
                emb, shape = na.flatten(neg)
                text_neg_embeds.append(emb)
                text_neg_shapes.append(shape)
        else:
            text_pos_embeds, text_pos_shapes = na.flatten(texts_pos)
            text_neg_embeds, text_neg_shapes = na.flatten(texts_neg)

        # Adapter les embeddings texte au dtype cible (compatible avec FP8)
        if isinstance(text_pos_embeds, torch.Tensor):
            text_pos_embeds = text_pos_embeds.to(target_dtype)
        if isinstance(text_neg_embeds, torch.Tensor):
            text_neg_embeds = text_neg_embeds.to(target_dtype)
        
        self.debug.log(f"Text embeddings adapted to precision: {target_dtype}", category="precision")
        
        # Flatten.
        latents, latents_shapes = na.flatten(noises)
        latents_cond, _ = na.flatten(conditions)

        # Adapter les latents au dtype cible (compatible avec FP8)
        latents = latents.to(target_dtype) if latents.dtype != target_dtype else latents
        latents_cond = latents_cond.to(target_dtype) if latents_cond.dtype != target_dtype else latents_cond

        
        if preserve_vram:
            if conditions[0].shape[0] > 1:
                self.debug.start_timer("vae_to_cpu")
                self.vae = self.vae.to("cpu")
                self.debug.end_timer("vae_to_cpu", "VAE to CPU")
            # Before sampling, check if BlockSwap is active
            if not use_blockswap and not hasattr(self, "_blockswap_active"):
                self.debug.start_timer("dit_to_gpu")
                self.dit = self.dit.to(get_device())
                self.debug.end_timer("dit_to_gpu", "DiT to GPU")
            else:
                # BlockSwap manages device placement
                pass

        self.debug.start_timer("dit_inference")
        
        with torch.autocast("cuda", target_dtype, enabled=True):
            latents = self.sampler.sample(
                x=latents,
                f=lambda args: classifier_free_guidance_dispatcher(
                    pos=lambda: self.dit(
                        vid=torch.cat([args.x_t, latents_cond], dim=-1),
                        txt=text_pos_embeds,
                        vid_shape=latents_shapes,
                        txt_shape=text_pos_shapes,
                        timestep=args.t.repeat(batch_size),
                    ).vid_sample,
                    neg=lambda: self.dit(
                        vid=torch.cat([args.x_t, latents_cond], dim=-1),
                        txt=text_neg_embeds,
                        vid_shape=latents_shapes,
                        txt_shape=text_neg_shapes,
                        timestep=args.t.repeat(batch_size),
                    ).vid_sample,
                    scale=(
                        cfg_scale
                        if (args.i + 1) / len(self.sampler.timesteps)
                        <= self.config.diffusion.cfg.get("partial", 1)
                        else 1.0
                    ),
                    rescale=self.config.diffusion.cfg.rescale,
                ),
            )
        
        self.debug.end_timer("dit_inference", "DiT inference completed")

        latents = na.unflatten(latents, latents_shapes)
        #self.debug.log(f"UNFLATTEN time: {time.time() - t} seconds", category="timing")
        
        # 🎯 Pré-calcul des dtypes (une seule fois)
        vae_dtype = getattr(torch, self.config.vae.dtype)
        decode_dtype = torch.float16 if (vae_dtype == torch.float16 or target_dtype == torch.float16) else vae_dtype
        self.debug.log(f"VAE decode precision: {decode_dtype}", category="precision")
        if preserve_vram:
            self.debug.start_timer("dit_to_cpu")
            self.dit = self.dit.to("cpu")
            latents_cond = latents_cond.to("cpu")
            latents_shapes = latents_shapes.to("cpu")
            if latents[0].shape[0] > 1:
                clear_vram_cache(self.debug)
            self.debug.end_timer("dit_to_cpu", "DiT moved to CPU")

            if latents[0].shape[0] > 1:
                self.debug.start_timer("vae_to_gpu")
                self.vae = self.vae.to(get_device())
                
                self.debug.end_timer("vae_to_gpu", "VAE moved to GPU")




        #with torch.autocast("cuda", decode_dtype, enabled=True):
        samples = self.vae_decode(latents, target_dtype=decode_dtype, preserve_vram=preserve_vram)
        
        self.debug.log(f"Samples shape: {samples[0].shape}", category="vae")
        #self.debug.log(f"🔄  ULTRA-FAST VAE DECODE time: {time.time() - t} seconds", category="timing")
        #t = time.time()
        #self.dit.to(get_device())
        #self.vae.to("cpu")
        #self.debug.log(f"🔄 Dit to GPU time: {time.time() - t} seconds", category="timing")
        #t = time.time()
        # 🚀 CORRECTION CRITIQUE: Conversion batch Float16 pour ComfyUI (plus rapide)
        if samples and len(samples) > 0 and samples[0].dtype != torch.float16:
            self.debug.log(f"Converting {len(samples)} samples from {samples[0].dtype} to Float16", category="precision")
            samples = [sample.to(torch.float16, non_blocking=True) for sample in samples]
        
        #self.debug.log(f"🚀 Conversion batch Float16 time: {time.time() - t} seconds", category="timing")
        
        # 🚀 OPTIMISATION: Nettoyage final minimal
        #t = time.time()
        #if dit_offload:
        #    self.vae.to("cpu")
        #    torch.cuda.empty_cache()
        #    self.dit.to(get_device())
        #else:
            # Garder VAE sur GPU pour les prochains appels
        #torch.cuda.empty_cache()
        #self.debug.log(f"🔄 FINAL CLEANUP time: {time.time() - t} seconds", category="timing")

        
        return samples
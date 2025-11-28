'''
configure VAE, DiT, text_enc etc.
'''

import torch

from opensora.models import CausalVAEModelWrapper
from opensora.models.text_encoder import get_text_enc, get_text_warpper
from opensora.models.causalvideovae import ae_stride_config, ae_channel_config
from opensora.models.causalvideovae import ae_norm, ae_denorm
from opensora.models.diffusion import Diffusion_models, Diffusion_models_class


class T2VBulidingBlocks():
    def __init__(self, args, weight_dtype):
        self.args = args
        self.weight_dtype = weight_dtype
        pass

    def define_vae(self):
        kwargs = {}
        args = self.args
        ae = CausalVAEModelWrapper(args.ae_path, cache_dir=args.cache_dir, **kwargs).eval()
        if args.enable_tiling:
            ae.vae.enable_tiling()
            ae.vae.tile_overlap_factor = args.tile_overlap_factor
        ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
        ae.vae_scale_factor = (ae_stride_t, ae_stride_h, ae_stride_w)
        assert ae_stride_h == ae_stride_w, f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
        args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
        args.ae_stride = args.ae_stride_h
        patch_size = args.model[-3:]
        patch_size_t, patch_size_h, patch_size_w = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
        args.patch_size = patch_size_h
        args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
        assert patch_size_h == patch_size_w, f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
        # assert args.num_frames % ae_stride_t == 0, f"Num_frames must be divisible by ae_stride_t, but found num_frames ({args.num_frames}), ae_stride_t ({ae_stride_t})."
        assert args.max_height % ae_stride_h == 0, f"Height must be divisible by ae_stride_h, but found Height ({args.max_height}), ae_stride_h ({ae_stride_h})."
        assert args.max_width % ae_stride_h == 0, f"Width size must be divisible by ae_stride_h, but found Width ({args.max_width}), ae_stride_h ({ae_stride_h})."
        args.stride_t = ae_stride_t * patch_size_t
        args.stride = ae_stride_h * patch_size_h
        latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
        ae.latent_size = latent_size
        if args.num_frames % 2 == 1:
            args.latent_size_t = latent_size_t = (args.num_frames - 1) // ae_stride_t + 1
        else:
            latent_size_t = args.num_frames // ae_stride_t
        
        self.ae = ae
        self.latent_size = latent_size
        self.latent_size_t = latent_size_t

    def define_text_enc(self):
        args = self.args
        kwargs = {'load_in_8bit': args.enable_8bit_t5, 'torch_dtype': weight_dtype, 'low_cpu_mem_usage': True}
        text_enc = get_text_warpper(args.text_encoder_name)(args, **kwargs).eval()
        self.text_enc = text_enc

    def define_dit(self):
        args = self.args
        model = Diffusion_models[args.model](
            in_channels=ae_channel_config[args.ae],
            out_channels=ae_channel_config[args.ae],
            # caption_channels=4096,
            # cross_attention_dim=1152,
            attention_bias=True,
            sample_size=self.latent_size,
            sample_size_t=self.latent_size_t,
            num_vector_embeds=None,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            use_linear_projection=False,
            only_cross_attention=False,
            double_self_attention=False,
            upcast_attention=False,
            # norm_type="ada_norm_single",
            norm_elementwise_affine=False,
            norm_eps=1e-6,
            attention_type='default',
            attention_mode=args.attention_mode,
            interpolation_scale_h=args.interpolation_scale_h,
            interpolation_scale_w=args.interpolation_scale_w,
            interpolation_scale_t=args.interpolation_scale_t,
            downsampler=args.downsampler,
            # compress_kv_factor=args.compress_kv_factor,
            use_rope=args.use_rope,
            # model_max_length=args.model_max_length,
            use_stable_fp32=args.enable_stable_fp32, 
        )
        model.gradient_checkpointing = args.gradient_checkpointing
        if args.pretrained:
            model_state_dict = model.state_dict()
            if 'safetensors' in args.pretrained:  # pixart series
                from safetensors.torch import load_file as safe_load
                # import ipdb;ipdb.set_trace()
                pretrained_checkpoint = safe_load(args.pretrained, device="cpu")
                pretrained_keys = set(list(pretrained_checkpoint.keys()))
                model_keys = set(list(model_state_dict.keys()))
                common_keys = list(pretrained_keys & model_keys)
                checkpoint = {k: pretrained_checkpoint[k] for k in common_keys if model_state_dict[k].numel() == pretrained_checkpoint[k].numel()}
            else:
                checkpoint = torch.load(args.pretrained, map_location='cpu')
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            logger.info(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
            logger.info(f'Successfully load {len(model_state_dict) - len(missing_keys)}/{len(model_state_dict)} keys from {args.pretrained}!')
        self.model = model

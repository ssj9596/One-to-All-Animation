from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

from opensora.controlnet_modules.controlnet import ControlNetConditioningEmbedding, zero_module, MiniHunyuanEncoder

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .attenion import attention, get_cu_seqlens
from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .token_refiner import SingleTokenRefiner

from .models_refextractor import MMDoubleStreamBlock, MMSingleStreamBlock
from .models_textfree import MMDoubleStreamBlock as MMDoubleStreamBlockTextFree
from .models_textfree import MMSingleStreamBlock as MMSingleStreamBlockTextFree
from .ipablocks import MMDoubleStreamBlockReFuser, MMSingleStreamBlockReFuser

from .utils import ckpt_kwargs

@dataclass
class ControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]
    controlnet_single_block_samples: Tuple[torch.Tensor]

class HYVideoDiffusionTransformerControlNet(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        text_states_dim,
        text_states_dim_2,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        rope_theta = 256,
        embedded_guidance_scale = 6.0, #用默认值
        conditioning_embedding_channels: int = None,
        add_ref_img = False,
        exclude_noise = False,
        text_free = False,
    ):  
        device = None
        dtype = None
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.use_refuser = False

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.embedded_guidance_scale = embedded_guidance_scale
        self.exclude_noise = exclude_noise
        

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )
        

        # zero init emb
        self.ref_img_in = zero_module(PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )) if add_ref_img else None

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size, **factory_kwargs
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )
        self.text_free = text_free
        if text_free:
            DoubleStreamBlock = MMDoubleStreamBlockTextFree
            SingleStreamBlock = MMSingleStreamBlockTextFree
        else:
            DoubleStreamBlock = MMDoubleStreamBlock
            SingleStreamBlock = MMSingleStreamBlock

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )
        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.double_blocks)):
            self.controlnet_blocks.append(zero_module(nn.Linear(self.hidden_size, self.hidden_size)))
        
        self.controlnet_single_blocks = nn.ModuleList([])
        for _ in range(len(self.single_blocks)):
            self.controlnet_single_blocks.append(zero_module(nn.Linear(self.hidden_size, self.hidden_size)))
        
        # TODO: refine
        # 仅限图像condition，直接过原来的block
        if not exclude_noise:
            # print("not exclude_noise")
            if conditioning_embedding_channels is not None: #输入可以用VAE进行压缩，这样不需要这个；当没有用VAE压缩，就需要这里用一个小网络进行额外的压缩
                # self.input_hint_block = ControlNetConditioningEmbedding(
                #     conditioning_embedding_channels=conditioning_embedding_channels,
                #     block_out_channels=(16, 16, 16, 16)
                # )

                self.input_hint_block = MiniHunyuanEncoder(
                    in_channels = 3, #一般肯定是3
                    out_channels = conditioning_embedding_channels, #一般情况下，和self.out_channels一致，这里主要是考虑可能用不同的patch_embedding方案
                    block_out_channels=(16, 16, 16, 16),
                    norm_num_groups = 4,
                    layers_per_block=1
                )
                self.controlnet_x_embedder = torch.nn.Linear(conditioning_embedding_channels*4, self.hidden_size)
            else:
                self.condition_in = zero_module(PatchEmbed(
                    self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
                ))
                self.input_hint_block = None
                self.controlnet_x_embedder = zero_module(torch.nn.Linear(self.in_channels, self.hidden_size))


        self.gradient_checkpointing = False


    def setup_refuser(self, weight_dtype):
        # only set once
        if self.text_free:
            print("text_free has not develop fuser!!!")
            return
        if self.use_refuser is True:
            return 
        from .ipablocks import MMSingleStreamBlockReFuser, MMDoubleStreamBlockReFuser
        if len(self.double_blocks) > 0:
            # ----  double_blocks ----
            new_double_blocks = []
            for i, block in enumerate(self.double_blocks):
                wrapped = MMDoubleStreamBlockReFuser(
                    original_block=block,
                ).to(dtype=weight_dtype)
                new_double_blocks.append(wrapped)
            self.double_blocks = nn.ModuleList(new_double_blocks)

        if len(self.single_blocks) > 0:
            new_single_blocks = []
            for i, block in enumerate(self.single_blocks):
                wrapped_sblock = MMSingleStreamBlockReFuser(
                    original_block=block,
                ).to(dtype=weight_dtype)
                new_single_blocks.append(wrapped_sblock)
            self.single_blocks = nn.ModuleList(new_single_blocks)
        self.use_refuser = True

    def enable_refuser_params_grad(self):
        if not self.use_refuser:
            print("[Refuser] Not use Refuser, skip enabling grad.")
            return

        for block in self.double_blocks:
            if hasattr(block, 'refuser') and block.refuser is not None:
                for p in block.refuser.parameters():
                    p.requires_grad = True
        for block in self.single_blocks:
            if hasattr(block, 'refuser') and block.refuser is not None:
                for p in block.refuser.parameters():
                    p.requires_grad = True

        print("[Refuser] All Refuser params set requires_grad=True.")

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()



    def forward(self, 
        hidden_states,
        controlnet_cond,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask,
        embed_cfg_scale,
        conditioning_scale = 1.0,
        return_dict = True,
        image_latents = None, 
        ref_block_samples = None,
        ref_single_block_samples = None,
        **kwargs, #用于后续传入guidance
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        lora_scale = 1.0 #TODO: 作为参数传入
        # 这个image_latents不要用了,一般都是None
        image_latents = None
        assert self.ref_img_in is None or image_latents is not None, "If self.ref_img_in is not None, image_latents must not be None."
        # 3.11 目前用不到这个!!
        repeat_t = hidden_states.shape[2]

        x = hidden_states
        t = timestep
        text_states, text_states_2 = encoder_hidden_states
        text_mask, test_mask_2 = encoder_attention_mask

        embedded_guidance_scale = embed_cfg_scale#TODO： maybe problem
        guidance = (
            torch.tensor(
                embedded_guidance_scale,
                dtype=torch.float32,
                device=hidden_states.device,
            ).to(hidden_states.dtype)
            * 1000.0
            if embedded_guidance_scale is not None
            else None
        )

        out = {}
        img = x if not self.exclude_noise else controlnet_cond
        txt = text_states
        _, _, ot, oh, ow = x.shape
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(ot, oh, ow)
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        # import pdb; pdb.set_trace()
        
        #TODO: fix
        if not self.exclude_noise:
            if self.input_hint_block is not None:
                controlnet_cond = self.input_hint_block(controlnet_cond)
                #注意这里是原始flux模型的patchify方案，和混元的并不一样，为了避免增加更多的参数，这里用这种pixel shuffle的方式进行patchify
                batch_size, channels, t_pw, height_pw, width_pw = controlnet_cond.shape
                patch_size_t, patch_size_h, patch_size_w = self.config.patch_size
                height = height_pw // patch_size_h
                width = width_pw // patch_size_w
                t_ = t_pw // patch_size_t
                controlnet_cond = controlnet_cond.reshape(
                    batch_size, channels, t_, patch_size_t, height, patch_size_h, width, patch_size_w
                )
                controlnet_cond = controlnet_cond.permute(0, 2, 4, 6, 1, 3, 5, 7)
                controlnet_cond = controlnet_cond.reshape(batch_size, height * width * t_, -1)

                img = img + self.controlnet_x_embedder(controlnet_cond)
            else:
                img = img + self.condition_in(controlnet_cond)
        
        # import pdb; pdb.set_trace()
        if self.ref_img_in:
            # -3,3
            image_latents = image_latents.repeat(1, 1, repeat_t, 1, 1)
            image_latents = self.ref_img_in(image_latents)
            # -0.7,0.7
            img = img + image_latents

        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        block_samples = ()
        for index, block in enumerate(self.double_blocks):
            ref_latent = None
            if ref_block_samples is not None:
                # 注意这里的ref_block_samples一定大于double_blocks
                interval_ref = len(ref_block_samples) / len(self.double_blocks) 
                ref_latent = ref_block_samples[int(min(index * interval_ref,len(ref_block_samples)-1))]

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                
                img, txt = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                    ref_latent,
                    **ckpt_kwargs,
                )
            else:
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                    ref_latent,
                ]

                img, txt = block(*double_block_args)
            block_samples = block_samples + (img, ) #NOTE: 这里仅仅处理了image模态数据

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        single_block_samples = ()
        if len(self.single_blocks) > 0:
            for index, block in enumerate(self.single_blocks):
                ref_latent = None
                if ref_single_block_samples is not None:
                    interval_ref = len(ref_single_block_samples) / len(self.single_blocks) 
                    ref_latent = ref_single_block_samples[int(min(index * interval_ref,len(ref_single_block_samples)-1))]

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                        ref_latent,
                        **ckpt_kwargs,
                    )
                else:
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                        ref_latent,
                    ]
                    x = block(*single_block_args)
                
                single_block_samples = single_block_samples + (x[:, :img.shape[1]], ) #NOTE: 只使用image模态
        
        controlnet_block_samples = ()
        for block_sample, controlnet_block in zip(block_samples, self.controlnet_blocks):
            block_sample = controlnet_block(block_sample)
            controlnet_block_samples = controlnet_block_samples + (block_sample, )
        
        controlnet_single_block_samples = ()
        for single_block_sample, controlnet_block in zip(single_block_samples, self.controlnet_single_blocks):
            single_block_sample = controlnet_block(single_block_sample)
            controlnet_single_block_samples = controlnet_single_block_samples + (single_block_sample,)
        
        controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
        controlnet_single_block_samples = [sample * conditioning_scale for sample in controlnet_single_block_samples]
        controlnet_block_samples = None if len(controlnet_block_samples) == 0 else controlnet_block_samples
        controlnet_single_block_samples = (
            None if len(controlnet_single_block_samples) == 0 else controlnet_single_block_samples
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_samples, controlnet_single_block_samples)

        return ControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )

    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        # 884 
        latents_size = [video_length, height, width]

        if isinstance(self.config.patch_size, int):
            assert all(s % self.config.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.config.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.config.patch_size for s in latents_size]
        elif isinstance(self.config.patch_size, list):
            assert all(
                s % self.config.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.config.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // self.config.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.config.hidden_size // self.config.heads_num
        rope_dim_list = self.config.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.config.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts
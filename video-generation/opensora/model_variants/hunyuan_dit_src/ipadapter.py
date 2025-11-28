import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
import numpy as np
from diffusers.models.normalization import RMSNorm
from einops import rearrange
# from flash_attn import flash_attn_func
try:
    # flash attn 3
    from flash_attn_interface import flash_attn_func
    # from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn_func = None
from .posemb_layers import apply_rotary_emb_k,apply_rotary_emb



class RefAttnProcessor_V2(nn.Module):
    def __init__(self, heads_num, hidden_size, cross_attention_dim=None, scale_end = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.heads_num = heads_num
        self.scale_end = scale_end
        
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        head_dim = hidden_size // heads_num
        self.attn_k_norm = RMSNorm(head_dim, eps=1e-6, elementwise_affine=True)

            
    def __call__(
        self,
        query,
        ref_latent: torch.FloatTensor,
        ref_freqs_cis,
        txt_len=0,
        spatialattn_include_txt: bool = False,
    ) -> torch.FloatTensor:

        scale = self.scale_end
        ref_k = self.to_k_ip(ref_latent)
        ref_v = self.to_v_ip(ref_latent)
        # import pdb; pdb.set_trace()

        ref_k = rearrange(ref_k, 'B L (H D) -> B L H D', H=self.heads_num)
        ref_v = rearrange(ref_v, 'B L (H D) -> B L H D', H=self.heads_num)

        ref_k = self.attn_k_norm(ref_k)

        # double block
        if txt_len == 0:
            img_q = query
            if ref_k.shape[1] != img_q.shape[1]:
                # img_q: [B, T*HW, H, D]
                B, L_q, H, D = img_q.shape
                L_k = ref_k.shape[1]
                T = L_q // L_k
                assert T * L_k == L_q, "img_q length must be divisible by spatial size"

                # reshape img_q -> [B*T, L_k, H, D]
                img_q = img_q.contiguous().view(B*T, L_k, H, D)

                # repeat ref_k/v -> [B*T,L_k, H,  D]
                ref_k = ref_k.repeat(1, T, 1, 1, 1).view(B * T, L_k, H, D)
                ref_v = ref_v.repeat(1, T, 1, 1, 1).view(B * T, L_k, H, D)

                img_q, ref_k = apply_rotary_emb(img_q, ref_k, freqs_cis=ref_freqs_cis, head_first=False)
                # img_q in double block
                attn,_ = flash_attn_func(
                    img_q.to(ref_k.device).to(ref_k.dtype),
                    ref_k,
                    ref_v,
                    softmax_scale=img_q.shape[-1]**-0.5,
                    causal=False,
                )
                attn = attn.view(B, L_q, H*D)
            else:
                # normal case
                img_q, ref_k = apply_rotary_emb(img_q, ref_k, freqs_cis=ref_freqs_cis, head_first=False)
                # # img_q in double block
                attn,_ = flash_attn_func(
                    img_q.to(ref_k.device).to(ref_k.dtype),
                    ref_k,
                    ref_v,
                    softmax_scale=img_q.shape[-1]**-0.5,
                    causal=False,
                )
                attn = rearrange(attn, "B L H D -> B L (H D)")
    

        # single block
        elif txt_len > 0:
            img_q, txt_q = query[:, :-txt_len, :, :], query[:, -txt_len:, :, :]
            # 2d 
            if ref_k.shape[1] != img_q.shape[1]:
                B, L_q, H, D = img_q.shape
                L_k = ref_k.shape[1]
                T = L_q // L_k
                assert T * L_k == L_q, "img_q length must be divisible by spatial size"

                # reshape img_q -> [B*T, L_k, H, D]
                img_q = img_q.contiguous().view(B*T, L_k, H, D)
                # repeat ref_k/v -> [B*T,L_k, H,  D]
                ref_k = ref_k.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, L_k, H, D)
                ref_v = ref_v.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, L_k, H, D)

                img_q, ref_k = apply_rotary_emb(img_q, ref_k, freqs_cis=ref_freqs_cis, head_first=False)

                if spatialattn_include_txt:
                    txt_q = txt_q.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, txt_len, H, D)
                    query = torch.cat([img_q, txt_q], dim=1)  # [B*T, HW+txt_len, H, D]
                else:
                    query = img_q
                # query in single block
                attn,_ = flash_attn_func(
                    query.to(ref_k.device).to(ref_k.dtype),
                    ref_k,
                    ref_v,
                    softmax_scale=query.shape[-1]**-0.5,
                    causal=False,
                )
                attn = attn.view(B, L_q, H*D)
            # 3d
            else:
                img_q, ref_k = apply_rotary_emb(img_q, ref_k, freqs_cis=ref_freqs_cis, head_first=False)
                query = torch.cat([img_q, txt_q], dim=1)  # [B, THW+txt_len, H, D]
                attn,_ = flash_attn_func(
                    query.to(ref_k.device).to(ref_k.dtype),
                    ref_k,
                    ref_v,
                    softmax_scale=query.shape[-1]**-0.5,
                    causal=False,
                )
                attn = rearrange(attn, "B L H D -> B L (H D)")

        attn = attn.to(img_q.dtype).to(img_q.device)
        return scale * attn


class RefAttnProcessor(nn.Module):
    def __init__(self, heads_num, hidden_size, cross_attention_dim=None, scale_end = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.heads_num = heads_num
        self.scale_end = scale_end
        
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        head_dim = hidden_size // heads_num
        self.attn_k_norm = RMSNorm(head_dim, eps=1e-6, elementwise_affine=True)

            
    def __call__(
        self,
        query,
        ref_latent: torch.FloatTensor,
        ref_freqs_cis,
        # img_q = None
    ) -> torch.FloatTensor:

        scale = self.scale_end
        ref_k = self.to_k_ip(ref_latent)
        ref_v = self.to_v_ip(ref_latent)
        # import pdb; pdb.set_trace()
        ref_k = rearrange(ref_k, 'B L (H D) -> B L H D', H=self.heads_num)
        ref_v = rearrange(ref_v, 'B L (H D) -> B L H D', H=self.heads_num)

        ref_k = self.attn_k_norm(ref_k)
        # query has been apply rotary emb
        # singleblock需要img_q
        # 4.5改为ref_k
        # import pdb; pdb.set_trace()
        ref_k = apply_rotary_emb_k(ref_k, ref_k, ref_freqs_cis, head_first=False)
        attn,_ = flash_attn_func(
                query.to(ref_latent.device).to(ref_latent.dtype),
                ref_k,
                ref_v,
                softmax_scale=query.shape[-1]**-0.5,
                causal=False,
            )
        attn = rearrange(attn, "B L H D -> B L (H D)")
        attn = attn.to(query.dtype).to(query.device)

        return scale * attn


# from IPAFluxAttnProcessor2_0Advanced
class HYVideoAttnProcessor2_0(nn.Module):
    _instances = set()
    _global_call_count = 0
    _last_timestep_printed = None
    _first_instance_for_timestep = None  # Add this line
    
    def __init__(self, num_tokens, hidden_size, cross_attention_dim=None, scale_start=1.0, scale_end=1.0, total_steps=1, timestep_range=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale_start = scale_start
        self.scale_end = scale_end
        self.total_steps = total_steps
        self.num_tokens = num_tokens
        
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)
        self.norm_added_v = RMSNorm(128, eps=1e-5, elementwise_affine=False)
        self.timestep_range = timestep_range

        self.seen_timesteps = set()
        self.steps = 0
        
        # Add this instance to the set of instances
        self.__class__._instances.add(self)
    
    def clear_memory(self):
        self.seen_timesteps.clear()
        if hasattr(self, 'to_k_ip'):
            del self.to_k_ip
        if hasattr(self, 'to_v_ip'):
            del self.to_v_ip
        if hasattr(self, 'norm_added_k'):
            del self.norm_added_k
        if hasattr(self, 'norm_added_v'):
            del self.norm_added_v

    @classmethod
    def reset_all_instances(cls):
        """Reset all instances of the class"""
        cls._global_call_count = 0
        cls._last_timestep_printed = None
        cls._first_instance_for_timestep = None  # Add this line
        for instance in cls._instances:
            instance.seen_timesteps.clear()
            instance.steps = 0

    def reset_steps(self):
        """Reset the steps counter and seen timesteps for this instance."""
        self.seen_timesteps.clear()
        self.steps = 0
        self.__class__._last_timestep_printed = None
        # print(f"Steps and seen timesteps have been reset for this instance.")

    def __del__(self):
        # Remove this instance from the set when it's deleted
        self.__class__._instances.remove(self)
            
    def __call__(
        self,
        num_heads,
        query,
        image_emb: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        current_timestep = t[0].item()
        
        # Reset steps when starting a new sequence (timestep = 1.0)
        if abs(current_timestep - 1.0) < 1e-6:
            self.reset_steps()
            
        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                return None
        
        # Only update steps and print for the first instance that sees this timestep
        if current_timestep not in self.seen_timesteps:
            self.seen_timesteps.add(current_timestep)
            self.steps += 1
            
            # Only print if this is the first instance for this timestep
            if self.__class__._first_instance_for_timestep is None:
                self.__class__._first_instance_for_timestep = self
            
            if (self.__class__._first_instance_for_timestep == self and 
                current_timestep != self.__class__._last_timestep_printed):
                current_step = min(self.steps, self.total_steps)
                if self.total_steps > 1:
                    scale = self.scale_start + (self.scale_end - self.scale_start) * (current_step - 1) / (self.total_steps - 1)
                else:
                    scale = self.scale_end
                    
                # print(f"Timestep: {current_timestep}, Step: {self.steps}/{self.total_steps}, Weight: {scale}")
                self.__class__._last_timestep_printed = current_timestep
        
        # Calculate scale for return value
        current_step = min(self.steps, self.total_steps)
        if self.total_steps > 1:
            # scale_end and scale_start to get the scale
            scale = self.scale_start + (self.scale_end - self.scale_start) * (current_step - 1) / (self.total_steps - 1)
        else:
            scale = self.scale_end
            
        ip_hidden_states = image_emb
        ip_hidden_states_key_proj = self.to_k_ip(ip_hidden_states)
        ip_hidden_states_value_proj = self.to_v_ip(ip_hidden_states)
        # import pdb; pdb.set_trace()
        ip_hidden_states_key_proj = rearrange(ip_hidden_states_key_proj, 'B L (H D) -> B H L D', H=num_heads)
        ip_hidden_states_value_proj = rearrange(ip_hidden_states_value_proj, 'B L (H D) -> B H L D', H=num_heads)

        ip_hidden_states_key_proj = self.norm_added_k(ip_hidden_states_key_proj)
        ip_hidden_states_value_proj = self.norm_added_v(ip_hidden_states_value_proj)

        ip_hidden_states = F.scaled_dot_product_attention(query.to(image_emb.device).to(image_emb.dtype), 
                                                        ip_hidden_states_key_proj, 
                                                        ip_hidden_states_value_proj, 
                                                        dropout_p=0.0, is_causal=False)

        ip_hidden_states = rearrange(ip_hidden_states, "B H L D -> B L (H D)", H=num_heads)
        ip_hidden_states = ip_hidden_states.to(query.dtype).to(query.device)

        return scale * ip_hidden_states
    

class MLPProjModelAdvanced(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

# from InstantXFluxIPAdapterModelAdvanced
# not use 
# class HYVideoDiffusionTransformerIPAdapter:
#     def __init__(self, image_encoder_path, ip_ckpt, device, num_tokens=4):
#         self.device = device
#         self.image_encoder_path = image_encoder_path
#         self.ip_ckpt = ip_ckpt
#         self.num_tokens = num_tokens
#         # load image encoder
#         self.image_encoder = SiglipVisionModel.from_pretrained(self.image_encoder_path).to(self.device, dtype=torch.float16)
#         self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
#         # state_dict
#         self.state_dict = torch.load(os.path.join(MODELS_DIR,self.ip_ckpt), map_location="cpu")
#         self.joint_attention_dim = 4096
#         self.hidden_size = 3072

#     def init_proj(self):
#         self.image_proj_model = MLPProjModelAdvanced(
#             cross_attention_dim=self.joint_attention_dim,
#             id_embeddings_dim=1152, 
#             num_tokens=self.num_tokens,
#         ).to(self.device, dtype=torch.float16)

#     def set_ip_adapter(self, hunyuan_model, weight_params, timestep_percent_range=(0.0, 1.0)):
#         weight_start, weight_end, steps = weight_params
#         s = hunyuan_model.model_sampling
#         percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
#         timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))
#         ip_attn_procs = {}
#         dsb_count = len(hunyuan_model.diffusion_model.double_blocks)
#         for i in range(dsb_count):
#             name = f"double_blocks.{i}"
#             ip_attn_procs[name] = HYVideoAttnProcessor2_0(
#                     hidden_size=self.hidden_size,
#                     cross_attention_dim=self.joint_attention_dim,
#                     num_tokens=self.num_tokens,
#                     scale_start=weight_start,
#                     scale_end=weight_end,
#                     total_steps=steps,
#                     timestep_range=timestep_range
#                 ).to(self.device, dtype=torch.float16)
#         ssb_count = len(hunyuan_model.diffusion_model.single_blocks)
#         for i in range(ssb_count):
#             name = f"single_blocks.{i}"
#             ip_attn_procs[name] = HYVideoAttnProcessor2_0(
#                     hidden_size=self.hidden_size,
#                     cross_attention_dim=self.joint_attention_dim,
#                     num_tokens=self.num_tokens,
#                     scale_start=weight_start,
#                     scale_end=weight_end,
#                     total_steps=steps,
#                     timestep_range=timestep_range
#                 ).to(self.device, dtype=torch.float16)
#         return ip_attn_procs
    
#     def load_ip_adapter(self, hunyuan_model, weight, timestep_percent_range=(0.0, 1.0)):
#         self.image_proj_model.load_state_dict(self.state_dict["image_proj"], strict=True)
#         ip_attn_procs = self.set_ip_adapter(hunyuan_model, weight, timestep_percent_range)
#         ip_layers = torch.nn.ModuleList(ip_attn_procs.values())
#         ip_layers.load_state_dict(self.state_dict["ip_adapter"], strict=True)
#         return ip_attn_procs

#     @torch.inference_mode()
#     def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
#         if pil_image is not None:
#             if isinstance(pil_image, Image.Image):
#                 pil_image = [pil_image]
#             clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
#             clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
#             clip_image_embeds = clip_image_embeds.to(dtype=torch.float16)
#         else:
#             clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
#         image_prompt_embeds = self.image_proj_model(clip_image_embeds)
#         return image_prompt_embeds
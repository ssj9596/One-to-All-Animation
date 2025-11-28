# https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux/blob/main/flux/layers.py

import torch
import torch.nn as nn
from typing import Tuple, Optional
from einops import rearrange

from .modulate_layers import ModulateDiT, modulate, apply_gate
from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .attenion import attention, get_cu_seqlens
from .ipadapter import HYVideoAttnProcessor2_0, RefAttnProcessor, RefAttnProcessor_V2




class MMDoubleStreamBlockReFuser_V2(nn.Module):

    def __init__(
        self,
        original_block: nn.Module,
    ):
        super().__init__()

        # self.original_block = original_block
        self.refuser = RefAttnProcessor_V2(
                heads_num=original_block.heads_num,
                hidden_size=original_block.hidden_size,
            )

        # copy from  original_block
        self.heads_num       = original_block.heads_num
        self.deterministic   = original_block.deterministic

        # img
        self.img_mod         = original_block.img_mod
        self.img_norm1       = original_block.img_norm1
        self.img_attn_qkv    = original_block.img_attn_qkv
        self.img_attn_q_norm = original_block.img_attn_q_norm
        self.img_attn_k_norm = original_block.img_attn_k_norm
        self.img_attn_proj   = original_block.img_attn_proj
        self.img_norm2       = original_block.img_norm2
        self.img_mlp         = original_block.img_mlp
        # txt
        self.txt_mod         = original_block.txt_mod
        self.txt_norm1       = original_block.txt_norm1
        self.txt_attn_qkv    = original_block.txt_attn_qkv
        self.txt_attn_q_norm = original_block.txt_attn_q_norm
        self.txt_attn_k_norm = original_block.txt_attn_k_norm
        self.txt_attn_proj   = original_block.txt_attn_proj
        self.txt_norm2       = original_block.txt_norm2
        self.txt_mlp         = original_block.txt_mlp



    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        ref_latent = None,
        ref_freqs_cis = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        apply ipadopter after 'img_attn' 。
        """
        assert ref_latent is not None, "ref_latent not be None"
        # ----------------------------------------------------------------
        #    original forward
        # ----------------------------------------------------------------

        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)# [B, L, 3 * hidden_size]
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed 
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Save img_q before applying 3D RoPE, for later use in refuser
        img_q_2d = img_q.clone()  # 👈

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=img_k.shape[0],
        )

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        # ----------------------------------------------------------------
        #    add IP-Adapter for img_attn
        # ----------------------------------------------------------------
        # IP-Adapter (num_heads, query, image_emb, t)
        #  - query: img_q (shape [B, L, H, D])
        #    IP-Adapter attention need [B, H, L, D]
        #  - image_emb: shape [B, tokens, hidden_size]
        #  - t: timestep
        # ----------------------------------------------------------------

        query = img_q_2d
        # freqs_cis_q = freqs_cis
        # import pdb; pdb.set_trace()
        ip_hidden_states = self.refuser(
            query=query,
            ref_latent=ref_latent,
            ref_freqs_cis=ref_freqs_cis
        )
        # print("Before addition:")
        # print("img_attn - max:", img_attn.max(), "min:", img_attn.min(), "mean:", img_attn.mean())
        # print("ip_hidden_states - max:", ip_hidden_states.max(), "min:", ip_hidden_states.min(), "mean:", ip_hidden_states.mean())
        if ip_hidden_states is not None:
            # [B, L, hidden_size]
            img_attn = img_attn + ip_hidden_states
        # print("After addition:")
        # print("img_attn - max:", img_attn.max(), "min:", img_attn.min(), "mean:", img_attn.mean())
        # ----------------------------------------------------------------
        #    finish IP-Adapter 
        # ----------------------------------------------------------------

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt



class MMSingleStreamBlockReFuser_V2(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
    ):
        super().__init__()

        # self.original_block = original_block
        self.refuser = RefAttnProcessor_V2(
                heads_num=original_block.heads_num,
                hidden_size=original_block.hidden_size,
            )

        self.heads_num = original_block.heads_num
        self.hidden_size = original_block.hidden_size
        self.deterministic = original_block.deterministic
        self.mlp_hidden_dim = original_block.mlp_hidden_dim
        self.scale = original_block.scale

        self.linear1 = original_block.linear1
        self.linear2 = original_block.linear2
        self.q_norm = original_block.q_norm
        self.k_norm = original_block.k_norm
        self.pre_norm = original_block.pre_norm
        self.mlp_act = original_block.mlp_act
        self.modulation = original_block.modulation

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        ref_latent = None,
        ref_freqs_cis = None,
    ) -> torch.Tensor:
        assert ref_latent is not None, "ref_latent not be None"

        # ----------------------------------------------------------------
        #    original forward
        # ----------------------------------------------------------------
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)


        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Save q before applying 3D RoPE, for later use in refuser
        q_2d = q.clone()  # 👈


        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=x.shape[0],
        )

        # attn => [B, L, hidden_size]
        # ------------------------------------------------------------------
        # add IP-Adapter
        # ------------------------------------------------------------------
        # use q instead of img_q
        # https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux/blob/main/flux/layers.py
        
        # q shape: [B, L, H, D]
        query = q_2d
        # freqs_cis_q = freqs_cis
        ip_hidden_states = self.refuser(
            query=query,
            ref_latent=ref_latent,
            ref_freqs_cis=ref_freqs_cis,
            txt_len=txt_len,
        )
        if ip_hidden_states is not None:
            # ip_hidden_states => [B, img_len, hidden_size]
            if ip_hidden_states.shape[1] < attn.shape[1]:
                # 2d case
                assert img_q.shape[1] == ip_hidden_states.shape[1]
                attn = torch.cat([
                    attn[:, :ip_hidden_states.shape[1]] + ip_hidden_states,
                    attn[:, ip_hidden_states.shape[1]:]
                ], dim=1)
            else:
                attn += ip_hidden_states

                
        # ----------------------------------------------------------------
        #    finish IP-Adapter 
        # ----------------------------------------------------------------

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)



class MMDoubleStreamBlockReFuser(nn.Module):

    def __init__(
        self,
        original_block: nn.Module,
    ):
        super().__init__()

        # self.original_block = original_block
        self.refuser = RefAttnProcessor(
                heads_num=original_block.heads_num,
                hidden_size=original_block.hidden_size,
            )

        # copy from  original_block
        self.heads_num       = original_block.heads_num
        self.deterministic   = original_block.deterministic

        # img
        self.img_mod         = original_block.img_mod
        self.img_norm1       = original_block.img_norm1
        self.img_attn_qkv    = original_block.img_attn_qkv
        self.img_attn_q_norm = original_block.img_attn_q_norm
        self.img_attn_k_norm = original_block.img_attn_k_norm
        self.img_attn_proj   = original_block.img_attn_proj
        self.img_norm2       = original_block.img_norm2
        self.img_mlp         = original_block.img_mlp
        # txt
        self.txt_mod         = original_block.txt_mod
        self.txt_norm1       = original_block.txt_norm1
        self.txt_attn_qkv    = original_block.txt_attn_qkv
        self.txt_attn_q_norm = original_block.txt_attn_q_norm
        self.txt_attn_k_norm = original_block.txt_attn_k_norm
        self.txt_attn_proj   = original_block.txt_attn_proj
        self.txt_norm2       = original_block.txt_norm2
        self.txt_mlp         = original_block.txt_mlp



    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        ref_latent = None,
        ref_freqs_cis = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        apply ipadopter after 'img_attn' 。
        """
        assert ref_latent is not None, "ref_latent not be None"
        # ----------------------------------------------------------------
        #    original forward
        # ----------------------------------------------------------------

        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)# [B, L, 3 * hidden_size]
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed 
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=img_k.shape[0],
        )

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        # ----------------------------------------------------------------
        #    add IP-Adapter for img_attn
        # ----------------------------------------------------------------
        # IP-Adapter (num_heads, query, image_emb, t)
        #  - query: img_q (shape [B, L, H, D])
        #    IP-Adapter attention need [B, H, L, D]
        #  - image_emb: shape [B, tokens, hidden_size]
        #  - t: timestep
        # ----------------------------------------------------------------

        query = img_q
        # freqs_cis_q = freqs_cis
        # import pdb; pdb.set_trace()
        ip_hidden_states = self.refuser(
            query=query,
            ref_latent=ref_latent,
            ref_freqs_cis=ref_freqs_cis
        )
        # print("Before addition:")
        # print("img_attn - max:", img_attn.max(), "min:", img_attn.min(), "mean:", img_attn.mean())
        # print("ip_hidden_states - max:", ip_hidden_states.max(), "min:", ip_hidden_states.min(), "mean:", ip_hidden_states.mean())
        if ip_hidden_states is not None:
            # [B, L, hidden_size]
            img_attn = img_attn + ip_hidden_states
        # print("After addition:")
        # print("img_attn - max:", img_attn.max(), "min:", img_attn.min(), "mean:", img_attn.mean())
        # ----------------------------------------------------------------
        #    finish IP-Adapter 
        # ----------------------------------------------------------------

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt



class MMSingleStreamBlockReFuser(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
    ):
        super().__init__()

        # self.original_block = original_block
        self.refuser = RefAttnProcessor(
                heads_num=original_block.heads_num,
                hidden_size=original_block.hidden_size,
            )

        self.heads_num = original_block.heads_num
        self.hidden_size = original_block.hidden_size
        self.deterministic = original_block.deterministic
        self.mlp_hidden_dim = original_block.mlp_hidden_dim
        self.scale = original_block.scale

        self.linear1 = original_block.linear1
        self.linear2 = original_block.linear2
        self.q_norm = original_block.q_norm
        self.k_norm = original_block.k_norm
        self.pre_norm = original_block.pre_norm
        self.mlp_act = original_block.mlp_act
        self.modulation = original_block.modulation

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        ref_latent = None,
        ref_freqs_cis = None,
    ) -> torch.Tensor:
        assert ref_latent is not None, "ref_latent not be None"

        # ----------------------------------------------------------------
        #    original forward
        # ----------------------------------------------------------------
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=x.shape[0],
        )

        # attn => [B, L, hidden_size]
        # ------------------------------------------------------------------
        # add IP-Adapter
        # ------------------------------------------------------------------
        # use q instead of img_q
        # https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux/blob/main/flux/layers.py
        
        # q shape: [B, L, H, D]
        query = q
        # freqs_cis_q = freqs_cis
        ip_hidden_states = self.refuser(
            query=query,
            ref_latent=ref_latent,
            ref_freqs_cis=ref_freqs_cis,
            # img_q=img_q,
        )
        if ip_hidden_states is not None:
            # ip_hidden_states => [B, img_len, hidden_size]
            attn = attn + ip_hidden_states
        # ----------------------------------------------------------------
        #    finish IP-Adapter 
        # ----------------------------------------------------------------

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)




class MMDoubleStreamBlockIPAdapter(nn.Module):

    def __init__(
        self,
        original_block: nn.Module,
        ipadapter_config: dict,
        ipadapter_scale: float = 1.0,
    ):
        super().__init__()

        # self.original_block = original_block
        self.ip_adapter = HYVideoAttnProcessor2_0(
                num_tokens=ipadapter_config["ipadapter_num_tokens"],
                hidden_size=ipadapter_config["ipadapter_hidden_size"],
                cross_attention_dim=ipadapter_config["ipadapter_cross_attention_dim"],
                # for scale
                scale_start=ipadapter_config["ipadapter_scale_start"],
                scale_end=ipadapter_config["ipadapter_scale_end"],
                total_steps=ipadapter_config["ipadapter_total_steps"],
            )
        self.ipadapter_scale = ipadapter_config['ipadapter_scale']

        # copy from  original_block
        self.heads_num       = original_block.heads_num
        self.deterministic   = original_block.deterministic

        # img
        self.img_mod         = original_block.img_mod
        self.img_norm1       = original_block.img_norm1
        self.img_attn_qkv    = original_block.img_attn_qkv
        self.img_attn_q_norm = original_block.img_attn_q_norm
        self.img_attn_k_norm = original_block.img_attn_k_norm
        self.img_attn_proj   = original_block.img_attn_proj
        self.img_norm2       = original_block.img_norm2
        self.img_mlp         = original_block.img_mlp
        # txt
        self.txt_mod         = original_block.txt_mod
        self.txt_norm1       = original_block.txt_norm1
        self.txt_attn_qkv    = original_block.txt_attn_qkv
        self.txt_attn_q_norm = original_block.txt_attn_q_norm
        self.txt_attn_k_norm = original_block.txt_attn_k_norm
        self.txt_attn_proj   = original_block.txt_attn_proj
        self.txt_norm2       = original_block.txt_norm2
        self.txt_mlp         = original_block.txt_mlp



    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        t: Optional[torch.Tensor] = None,
        image_emb = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        apply ipadopter after 'img_attn' 。
        """
        assert image_emb is not None, "image_emb not be None"
        # ----------------------------------------------------------------
        #    original forward
        # ----------------------------------------------------------------

        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)# [B, L, 3 * hidden_size]
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed 
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=img_k.shape[0],
        )

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        # ----------------------------------------------------------------
        #    add IP-Adapter for img_attn
        # ----------------------------------------------------------------
        # IP-Adapter (num_heads, query, image_emb, t)
        #  - query: img_q (shape [B, L, H, D])   instead of img_attn q
        #    IP-Adapter attention need [B, H, L, D]
        #  - image_emb: shape [B, tokens, hidden_size]
        #  - t: timestep
        # ----------------------------------------------------------------

        #  img_q  => [B, H, L, D]
        img_q_for_ip = img_q.transpose(1, 2)

        ip_hidden_states = self.ip_adapter(
            num_heads=self.heads_num,
            query=img_q_for_ip,
            image_emb=image_emb,
            t=t,
        )
        if ip_hidden_states is not None:
            # [B, L, hidden_size]
            img_attn = img_attn + self.ipadapter_scale * ip_hidden_states
        # ----------------------------------------------------------------
        #    finish IP-Adapter 
        # ----------------------------------------------------------------

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt



class MMSingleStreamBlockIPAdapter(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        ipadapter_config: dict,
        ipadapter_scale: float = 1.0,
    ):
        super().__init__()

        # self.original_block = original_block
        self.ip_adapter = HYVideoAttnProcessor2_0(
                num_tokens=ipadapter_config["ipadapter_num_tokens"],
                hidden_size=ipadapter_config["ipadapter_hidden_size"],
                cross_attention_dim=ipadapter_config["ipadapter_cross_attention_dim"],
                # for scale
                scale_start=ipadapter_config["ipadapter_scale_start"],
                scale_end=ipadapter_config["ipadapter_scale_end"],
                total_steps=ipadapter_config["ipadapter_total_steps"],
            )
        self.ipadapter_scale = ipadapter_config['ipadapter_scale']

        self.heads_num = original_block.heads_num
        self.hidden_size = original_block.hidden_size
        self.deterministic = original_block.deterministic
        self.mlp_hidden_dim = original_block.mlp_hidden_dim
        self.scale = original_block.scale

        self.linear1 = original_block.linear1
        self.linear2 = original_block.linear2
        self.q_norm = original_block.q_norm
        self.k_norm = original_block.k_norm
        self.pre_norm = original_block.pre_norm
        self.mlp_act = original_block.mlp_act
        self.modulation = original_block.modulation

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,  # <-- timestep
        image_emb = None,
    ) -> torch.Tensor:
        assert image_emb is not None, "image_emb not be None"

        # ----------------------------------------------------------------
        #    original forward
        # ----------------------------------------------------------------
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=x.shape[0],
        )

        # attn => [B, L, hidden_size]
        # ------------------------------------------------------------------
        # add IP-Adapter
        # ------------------------------------------------------------------
        # use q instead of img_q
        # https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux/blob/main/flux/layers.py
        
        # q shape: [B, L, H, D]
        img_q_for_ip = q.transpose(1, 2)
        
        ip_hidden_states = self.ip_adapter(
            num_heads=self.heads_num,
            query=img_q_for_ip,
            image_emb=image_emb,
            t=t,  # timestep
        )
        if ip_hidden_states is not None:
            # ip_hidden_states => [B, img_len, hidden_size]
            attn = attn + self.ipadapter_scale * ip_hidden_states
        # ----------------------------------------------------------------
        #    finish IP-Adapter 
        # ----------------------------------------------------------------

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)







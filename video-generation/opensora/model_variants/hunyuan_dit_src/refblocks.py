import torch
import torch.nn as nn
from typing import Tuple, Optional
from einops import rearrange

from .modulate_layers import ModulateDiT, modulate, apply_gate
from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .attenion import attention, get_cu_seqlens
from .ipadapter import HYVideoAttnProcessor2_0
from .mlp_layers import MLP
from .activation_layers import get_activation_layer
class MMDoubleStreamBlockRef(nn.Module):

    def __init__(
        self,
        original_block: nn.Module,
        hidden_size: int,
        mlp_act_type: str = "gelu_tanh",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ref_mlp = MLP(
            hidden_size,
            hidden_size,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.ref_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
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
        ref = None,
        text_mask = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        apply ipadopter after 'img_attn' 。
        """
        img_len = img.shape[1]
        txt_len = txt.shape[1]
        ref_len = ref.shape[1]

        cu_seqlens_q = get_cu_seqlens(text_mask, img_len + ref_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_len + ref_len + txt_len
        max_seqlen_kv = max_seqlen_q

        assert ref is not None, "ref not be None"
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
        # add ref here
        ref = self.ref_mlp(ref)
        ref = self.ref_norm(ref)

        img_branch_input = torch.cat([img_modulated, ref], dim=1)
        img_ref_qkv = self.img_attn_qkv(img_branch_input)# [B, L, 3 * hidden_size]
        img_ref_q, img_ref_k, img_ref_v = rearrange(
            img_ref_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed 
        img_ref_q = self.img_attn_q_norm(img_ref_q).to(img_ref_v)
        img_ref_k = self.img_attn_k_norm(img_ref_k).to(img_ref_v)

        
        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, ref_q = img_ref_q[:, :img_len, :, :], img_ref_q[:, img_len:, :, :]
            img_k, ref_k = img_ref_k[:, :img_len, :, :], img_ref_k[:, img_len:, :, :]

            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            img_ref_q = torch.cat([img_q, ref_q],dim=1)
            img_ref_k = torch.cat([img_k, ref_k],dim=1)

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
        q = torch.cat((img_ref_q, txt_q), dim=1)
        k = torch.cat((img_ref_k, txt_k), dim=1)
        v = torch.cat((img_ref_v, txt_v), dim=1)
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

        img_attn, txt_attn = attn[:, : img_len], attn[:, img_len+ref_len :]

        # # ----------------------------------------------------------------
        # #    add IP-Adapter for img_attn
        # # ----------------------------------------------------------------
        # # IP-Adapter (num_heads, query, image_emb, t)
        # #  - query: img_q (shape [B, L, H, D])   instead of img_attn q
        # #    IP-Adapter attention need [B, H, L, D]
        # #  - image_emb: shape [B, tokens, hidden_size]
        # #  - t: timestep
        # # ----------------------------------------------------------------

        # #  img_q  => [B, H, L, D]
        # img_q_for_ip = img_q.transpose(1, 2)

        # ip_hidden_states = self.ip_adapter(
        #     num_heads=self.heads_num,
        #     query=img_q_for_ip,
        #     image_emb=image_emb,
        #     t=t,
        # )
        # if ip_hidden_states is not None:
        #     # [B, L, hidden_size]
        #     img_attn = img_attn + self.ipadapter_scale * ip_hidden_states
        # # ----------------------------------------------------------------
        # #    finish IP-Adapter 
        # # ----------------------------------------------------------------

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



class MMSingleStreamBlockRef(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        hidden_size: int,
        mlp_act_type: str = "gelu_tanh",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ref_mlp = MLP(
            hidden_size,
            hidden_size,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.ref_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)

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
        ref = None,
        img_len = None,
        text_mask = None,
    ) -> torch.Tensor:
        assert ref is not None, "ref not be None"

        ref_len = ref.shape[1]

        cu_seqlens_q = get_cu_seqlens(text_mask, img_len + ref_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_len + ref_len + txt_len
        max_seqlen_kv = max_seqlen_q


        # ----------------------------------------------------------------
        #    original forward
        # ----------------------------------------------------------------
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)

        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

        # add ref here
        # ref_len = ref.shape[1]
        ref = self.ref_mlp(ref)
        ref = self.ref_norm(ref)
        x_mod = torch.cat([x_mod[:,:img_len],ref,x_mod[:,img_len:]],dim=1)

        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, ref_q, txt_q,  = q[:, :img_len, :, :], q[:, img_len:img_len+ref_len, :, :], q[:, img_len+ref_len:, :, :]
            img_k, ref_k, txt_k,  = k[:, :img_len, :, :], k[:, img_len:img_len+ref_len, :, :], k[:, img_len+ref_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat([img_q, ref_q, txt_q], dim=1)
            k = torch.cat([img_k, ref_k, txt_k], dim=1)

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
        attn = torch.cat([attn[:, : img_len],attn[:, img_len+ref_len :]],dim=1)

        # Compute activation in mlp stream, cat again and run second linear layer.
        mlp = torch.cat([mlp[:, : img_len],mlp[:, img_len+ref_len :]],dim=1)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)
    



# ================================================

# K V only

# ================================================


# Q not include ref
class MMDoubleStreamBlockRefKVOnly(nn.Module):

    def __init__(
        self,
        original_block: nn.Module,
        hidden_size: int,
        mlp_act_type: str = "gelu_tanh",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ref_mlp = MLP(
            hidden_size,
            hidden_size,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.ref_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
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
        ref = None,
        text_mask = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        apply ipadopter after 'img_attn' 。
        """
        img_len = img.shape[1]
        txt_len = txt.shape[1]
        ref_len = ref.shape[1]
        # import pdb
        # pdb.set_trace()
        # cu_seqlens_q and max_seqlen_q remain the input
        cu_seqlens_kv = get_cu_seqlens(text_mask, img_len + ref_len)
        max_seqlen_kv = img_len + ref_len + txt_len

        assert ref is not None, "ref not be None"
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

        # print("Double Block:=================================================")
        # print("before norm: img max:", img.max().item(), "img min:", img.min().item()) 
        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        # print("after norm: img_modulated max:", img_modulated.max().item(), "img_modulated min:", img_modulated.min().item()) 
        # add ref here
        # print("before mlp: ref max:", ref.max().item(), "ref min:", ref.min().item()) 
        ref = self.ref_mlp(ref)
        # print("after mlp: ref max:", ref.max().item(), "ref min:", ref.min().item()) 
        ref = self.ref_norm(ref)
        # print("after norm: ref max:", ref.max().item(), "ref min:", ref.min().item()) 

        img_branch_input = torch.cat([img_modulated, ref], dim=1)
        img_ref_qkv = self.img_attn_qkv(img_branch_input)# [B, L, 3 * hidden_size]
        img_ref_q, img_ref_k, img_ref_v = rearrange(
            img_ref_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed 
        img_q = img_ref_q[:, :img_len, :, :]
        img_q = self.img_attn_q_norm(img_q).to(img_ref_v)
        img_ref_k = self.img_attn_k_norm(img_ref_k).to(img_ref_v)

        
        # Apply RoPE if needed.
        if freqs_cis is not None:
        
            img_k, ref_k = img_ref_k[:, :img_len, :, :], img_ref_k[:, img_len:, :, :]
            # print("after attention&norm img_k max:", img_k.max().item(), "after attention&norm img_k min:", img_k.min().item())
            # print("after attention&norm ref_k max:", ref_k.max().item(), "after attention&norm ref_k min:", ref_k.min().item())
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

            img_ref_k = torch.cat([img_k, ref_k],dim=1)

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
        k = torch.cat((img_ref_k, txt_k), dim=1)
        v = torch.cat((img_ref_v, txt_v), dim=1)
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
        img_attn, txt_attn = attn[:, : img_len], attn[:, img_len :]
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



class MMSingleStreamBlockRefKVOnly(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        hidden_size: int,
        mlp_act_type: str = "gelu_tanh",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ref_mlp = MLP(
            hidden_size,
            hidden_size,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.ref_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)

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
        ref = None,
        img_len = None,
        text_mask = None,
    ) -> torch.Tensor:
        assert ref is not None, "ref not be None"
        # import pdb
        # pdb.set_trace()
        ref_len = ref.shape[1]
        
        
        # cu_seqlens_q and max_seqlen_q remain the input
        cu_seqlens_kv = get_cu_seqlens(text_mask, img_len + ref_len)
        max_seqlen_kv = img_len + ref_len + txt_len


        # ----------------------------------------------------------------
        #    original forward
        # ----------------------------------------------------------------
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        # print("Single Block:=================================================")
        # print("before norm: x max:", x.max().item(), "x min:", x.min().item()) 
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        # print("after norm: x_mod max:", x_mod.max().item(), "x min:", x_mod.min().item()) 

        # add ref here
        # ref_len = ref.shape[1]
        # print("before mlp_max: ref max:", ref.max().item(), "ref min:", ref.min().item()) 
        ref = self.ref_mlp(ref)
        # print("after mlp_max: ref max:", ref.max().item(), "ref min:", ref.min().item()) 
        ref = self.ref_norm(ref)
        # print("after norm: ref max:", ref.max().item(), "ref min:", ref.min().item()) 
        x_mod = torch.cat([x_mod[:,:img_len], ref, x_mod[:,img_len:]],dim=1)

        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        
        q = torch.cat([q[:, : img_len],q[:, img_len+ref_len:]],dim=1)
        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q  = q[:, :img_len, :, :], q[:, img_len:, :, :]
            img_k, ref_k, txt_k,  = k[:, :img_len, :, :], k[:, img_len:img_len+ref_len, :, :], k[:, img_len+ref_len:, :, :]
            # print("after linear1&norm img_k max:", img_k.max().item(), "after linear1&norm img_k min:", img_k.min().item())
            # print("after linear1&norm ref_k max:", ref_k.max().item(), "after linear1&norm ref_k min:", ref_k.min().item())
            # print("after linear1&norm txt_k max:", txt_k.max().item(), "after linear1&norm txt_k min:", txt_k.min().item())
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat([img_q, txt_q], dim=1)
            k = torch.cat([img_k, ref_k, txt_k], dim=1)

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

        # Compute activation in mlp stream, cat again and run second linear layer.
        mlp = torch.cat([mlp[:, : img_len],mlp[:, img_len+ref_len :]],dim=1)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)
    

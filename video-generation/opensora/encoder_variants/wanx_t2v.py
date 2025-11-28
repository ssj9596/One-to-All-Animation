import torch
import torch.nn as nn
from transformers import AutoTokenizer, UMT5EncoderModel
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
class WanxEncoderWrapperT2V(nn.Module):
    def __init__(self, model_path, weight_dtype):
        super().__init__()
        self.text_encoder = UMT5EncoderModel.from_pretrained(model_path, torch_dtype=weight_dtype, subfolder='text_encoder').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype=weight_dtype, subfolder = 'tokenizer')
        self.text_encoder.requires_grad_(False)
        # Load null text embeddings if available
        self.null_text_embedding = None
        null_embed_path = os.path.join(model_path, 'null_text_embedding.pt')
        if os.path.exists(null_embed_path):
            self.null_text_embedding = torch.load(null_embed_path).to(dtype=weight_dtype)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        return prompt_embeds

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):


        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds



    def forward(self, caption, device=None, max_length=226):
        if caption == "" or [""] and self.null_text_embedding is not None:
            return self.null_text_embedding.to(device=device)
        caption = [caption] if isinstance(caption, str) else caption
        cond = self.encode_prompt(caption, device= device, max_sequence_length=max_length)
        return cond

def save_null_text_embedding(model_path, max_length=226):
    """Save null text embeddings for empty strings"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Initialize model
    model = WanxEncoderWrapperT2V(model_path, weight_dtype=dtype)
    model.to(device)
    
    # Get embeddings for empty string
    null_embedding = model.encode_prompt(
        prompt="",
        max_sequence_length=max_length,
        device=device,
        dtype=dtype
    )
    print("null_embedding.shape",null_embedding.shape)
    # import pdb;pdb.set_trace()
    # Save the embeddings
    null_embedding = null_embedding.cpu()
    save_path = os.path.join(model_path, 'null_text_embedding.pt')
    torch.save(null_embedding, save_path)
    print(f"Saved null text embeddings to {save_path}")

if __name__ == "__main__":
    model_path = "../pretrained_models/Wan2.1-T2V-1.3B-Diffusers/"
    save_null_text_embedding(model_path)
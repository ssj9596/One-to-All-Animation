import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer

class T5EncoderWrapper(nn.Module):
    def __init__(self, model_path, weight_dtype):
        super().__init__()
        self.text_encoder = T5EncoderModel.from_pretrained(model_path, torch_dtype=weight_dtype, subfolder='text_encoder').eval()
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, torch_dtype=weight_dtype, subfolder = 'tokenizer')
        self.text_encoder.requires_grad_(False)

    def forward(self, caption, device=None, max_length=226):
        caption = [caption] if isinstance(caption, str) else caption
        text_inputs = self.tokenizer(
            caption, #convert to list if not
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)
        cond = self.text_encoder(text_input_ids.to(device))[0]
        cond_mask = prompt_attention_mask
        pooled_projections = None
        return cond, cond_mask, pooled_projections
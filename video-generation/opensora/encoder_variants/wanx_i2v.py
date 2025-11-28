import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from torchvision.transforms.functional import to_pil_image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from opensora.utils.utils import center_square_crop
class WanxEncoderWrapperI2V(nn.Module):
    def __init__(self, model_path, weight_dtype):
        super().__init__()
        self.text_encoder = UMT5EncoderModel.from_pretrained(model_path, torch_dtype=weight_dtype, subfolder='text_encoder').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype=weight_dtype, subfolder = 'tokenizer')
        self.text_encoder.requires_grad_(False)
        self.image_encoder = CLIPVisionModel.from_pretrained(
            model_path, subfolder="image_encoder", torch_dtype=weight_dtype
        ).eval()
        self.image_encoder.requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_path, subfolder="image_processor"
        )
    
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


    def encode_image(
        self,
        image,
        device: Optional[torch.device] = None,
    ):
        device = device

        # import pdb; pdb.set_trace()
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

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



    def forward(self, caption, image, device=None, max_length=226):
        # make sure image in Tensor [-1,1]
        caption = [caption] if isinstance(caption, str) else caption
        cond = self.encode_prompt(caption, device= device, max_sequence_length=max_length)
        image = to_pil_image(((image + 1) / 2.).clamp(0, 1))
        # image_crop = 
        croped_image = center_square_crop(image)
        image_cond = self.encode_image(croped_image, device=device)
        return cond, image_cond, croped_image
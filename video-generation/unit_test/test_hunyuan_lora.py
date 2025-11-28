import torch
from opensora.model_variants.hunyuan_dit_src import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
import safetensors

def load_state_dict_from_dir(target_dir):
    from safetensors.torch import load_file as safe_load
    shard_files = [f for f in os.listdir(target_dir) if f.endswith('.safetensors')]
    checkpoint = {}
    for shard_file in shard_files:
        state_dict = safe_load(os.path.join(target_dir, shard_file), device='cpu')
        checkpoint.update(state_dict)
    return checkpoint

config = HYVideoDiffusionTransformer.load_config("configs/hunyuan_i2v.json")
model = HYVideoDiffusionTransformer.from_config(config)

transformer_lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    init_lora_weights=True,
    target_modules=["img_attn_qkv", "img_attn_proj", "img_mlp.fc1", "img_mlp.fc2", \
        "linear1", "linear2"],
)

import ipdb
ipdb.set_trace()


model.requires_grad_(False)
model.add_adapter(transformer_lora_config)

transformer_lora_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
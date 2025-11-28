import os
import glob
import torch
import numpy as np
from opensora.sample.pipeline_wanx_vhuman_tokenreplace import WanPipeline
from opensora.model_variants.wanx_diffusers_src import WanTransformer3DModel_Refextractor_2D
from opensora.encoder_variants import get_text_enc
from opensora.vae_variants import get_vae
from datetime import datetime
from einops import rearrange
import imageio
from functools import partial
import traceback
from PIL import Image

from diffusers.utils import export_to_video, load_image
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# ===== Model Config =====
model_path = "../pretrained_models/Wan2.1-T2V-1.3B-Diffusers"
vae_path = "../pretrained_models/Wan2.1-T2V-1.3B-Diffusers/vae"
config_path = "configs/wan2.1_t2v_1.3b.json"

model_dtype = torch.bfloat16

# Scheduler
scheduler = FlowMatchEulerDiscreteScheduler(
    shift=7.0,
    num_train_timesteps=1000,
    use_dynamic_shifting=False
)

# VAE and Text Encoder
vae = get_vae('wanx', vae_path, model_dtype)
encoders = get_text_enc('wanx-t2v', model_path, model_dtype)
text_encoder = encoders.text_encoder
tokenizer = encoders.tokenizer

# Model
model = WanTransformer3DModel_Refextractor_2D.from_config(config_path).to(model_dtype)
model.set_up_refextractor("configs/wan2.1_t2v_1.3b_refextractor_2d_withmask2.json", model_dtype)
model.eval()
model.requires_grad_(False)

# Load checkpoint
def load_state_dict(ckpt_dir):
    from safetensors.torch import load_file as safe_load
    shard_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.safetensors')]
    checkpoint = {}
    for shard_file in shard_files:
        state_dict = safe_load(os.path.join(ckpt_dir, shard_file), device='cpu')
        checkpoint.update(state_dict)
    return checkpoint

# change your own path
ckpt_path = "./outputs_wanx1.3b/train1.3b_only_refextractor_2d_2/checkpoint-21366-6788/fp32_model_21366"
checkpoint = load_state_dict(ckpt_path)
model.load_state_dict(checkpoint, strict=True)

# Pipeline
pipe = WanPipeline(
    transformer=model,
    vae=vae.vae, 
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler
)
pipe.to("cuda", dtype=model_dtype)

# ===== Helper Functions =====
def resizecrop(image, th, tw):
    w, h = image.size
    if h / w > th / tw:
        new_w = int(w)
        new_h = int(new_w * th / tw)
    else:
        new_h = int(h)
        new_w = int(new_h * tw / th)
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    image = image.crop((left, top, right, bottom))
    image = image.resize((tw, th))
    return image

def find_cache_dir(cache_dir_root, ref_path, vid_path, alignmode):
    """Find cache directory based on reference, video and alignment mode"""
    refname = os.path.splitext(os.path.basename(ref_path))[0]
    vidname = os.path.splitext(os.path.basename(vid_path))[0]
    job_name = f"ref_{refname}_vid_{vidname}"
    pattern = os.path.join(
        cache_dir_root,
        f"ref_{refname}_driven_{vidname}_align_{alignmode}_*"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"❌ Cache directory not found: {pattern}")
    if len(matches) > 1:
        print(f"⚠ Multiple cache directories found, using first: {matches}")
    return matches[0], job_name

# ===== Inference Parameters =====
image_guidance_scale = 2.0
pose_guidance_scale = 0
black_pose_cfg = False
black_image_cfg = True
zero_neg = False
negative_prompt = "" if zero_neg else [
    "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
]

controlnet_conditioning_scale = 0  # No pose
num_frames = 81 # training 29 but inference 81
case1 = False

# Cache directory
cache_base_dir = "../input_cache"
# Output directory
output_base_dir = "../output/ref_only_1.3b"
# stage 1 only trained on 512px
max_short = 384 
# ===== Task List =====
# Task list format: (ref_path, vid_path, alignmode, new_h, new_w, ref_cfg, pose_cfg, prompt)
# new_h, new_w, ref_cfg, pose_cfg: "" means use default
task_list = [
    ("../examples/img.png", "../examples/vid.mp4", "ref", "", "", "", "", ""),
    ("../examples/joker2_resize.png", "../examples/douyinvid5_v2.mp4", "pose", "", "", "", "", "a clown in a vibrant red suit dancing joyfully"),
    ("../examples/musk.jpg", "../examples/vid2.mp4", "ref", "", "", "", "", ""),
    ("../examples/maodie.png", "../examples/vid2.mp4", "ref", "", "", "", "", ""),
]

# ===== Main Inference Loop =====
for ref_path, vid_path, alignmode, new_h, new_w, ref_cfg, pose_cfg, prompt in task_list:
    try:
        if not ref_cfg:
            ref_cfg = image_guidance_scale
        else:
            ref_cfg = float(ref_cfg)
        
        if not pose_cfg:
            pose_cfg = pose_guidance_scale
        else:
            pose_cfg = float(pose_cfg)
        
        try:
            cache_dir, job_name = find_cache_dir(cache_base_dir, ref_path, vid_path, alignmode)
        except FileNotFoundError as e:
            print(e)
            continue
        
        print(f"Processing: {job_name}")
        print(f"Using cache: {cache_dir}")
        print(f"Prompt: {prompt}")
        
        # Read from cache
        image_input = Image.open(os.path.join(cache_dir, "image_input.png")).convert("RGB")
        mask_input = Image.open(os.path.join(cache_dir, "mask_input.png")).convert("L")
        
        # Get dimensions
        if new_h and new_w:
            new_h = int(float(new_h))
            new_w = int(float(new_w))
        else:
            new_h, new_w = image_input.height, image_input.width
        
        if min(new_h, new_w) > max_short:
            if new_h < new_w:
                scale = max_short / new_h
                new_h, new_w = max_short, int(new_w * scale)
            else:
                scale = max_short / new_w
                new_w, new_h = max_short, int(new_h * scale)
                

        
        # Ensure dimensions are multiples of 16
        new_h = int(new_h // 16 * 16)
        new_w = int(new_w // 16 * 16)
        
        transform = partial(resizecrop, th=new_h, tw=new_w)
        
        # Apply transform
        image_input = transform(image_input)
        mask_input = transform(mask_input)
        
        # Prepare mask tensor
        mask_np = np.array(mask_input, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).unsqueeze(2)  # [1,1,1,H,W]
        
        
        # Prepare output path
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_dir = os.path.join(output_base_dir, f"{current_date}_ref_{ref_cfg}_pose_{pose_cfg}_frame_{num_frames}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{job_name}_frames{num_frames}.mp4")
        
        if os.path.exists(output_path):
            print(f"✓ {output_path} exists, skipping!")
            continue
        
        # Save input image for reference
        image_input.save(output_path.replace(".mp4", "_input.png"))
        
        print(f"Generating video...")
        
        # Inference - only using reference image, no control video
        output = pipe(
            image=image_input, 
            image_mask=mask_tensor,
            control_video=None,  # No pose control
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            height=new_h, 
            width=new_w, 
            num_frames=num_frames,
            image_guidance_scale=ref_cfg,
            pose_guidance_scale=pose_cfg,
            num_inference_steps=30,
            generator=torch.Generator(device="cuda").manual_seed(42),
            black_image_cfg=black_image_cfg,
            black_pose_cfg=black_pose_cfg,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            return_tensor=True,
            case1=case1,
            image_pose=None, 
        ).frames[0]
        
        # Save video
        pred = output.detach().cpu()  # (T,C,H,W)
        
        # Convert to video format
        images = (pred / 2 + 0.5).clamp(0, 1)
        img_tensors = rearrange(images.permute(1, 0, 2, 3) * 255, 't c h w -> t h w c').byte().numpy()
        
        imageio.mimwrite(
            output_path, 
            img_tensors,
            fps=30, 
            quality=8, 
            output_params=["-loglevel", "error"]
        )
        
        print(f"✓ Saved to: {output_path}\n")
        
    except Exception as e:
        print(f"❌ Error processing {ref_path}: {e}")
        traceback.print_exc()
        print()

print("All tasks completed!")

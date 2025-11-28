import os
import glob
import torch
import decord
import imageio
from PIL import Image
import torch.multiprocessing as mp
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
from functools import partial
from tqdm import tqdm
from datetime import datetime

import numpy as np
from safetensors.torch import load_file as safe_load
from opensora.sample.pipeline_wanx_vhuman_tokenreplace import WanPipeline
from opensora.model_variants.wanx_diffusers_src import WanTransformer3DModel_Refextractor_2D_Controlnet_prefix
from opensora.encoder_variants import get_text_enc
from opensora.vae_variants import get_vae
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# ===== Config =====
model_path = "../pretrained_models/Wan2.1-T2V-14B-Diffusers"
vae_path = "../pretrained_models/Wan2.1-T2V-14B-Diffusers/vae"
config_path = "configs/wan2.1_t2v_14b.json"
model_dtype = torch.bfloat16

ckpt_paths = [
    "../checkpoints/One-to-All-14b",
]
default_cfg_combos = [
    (2.0, 1.5), # 3x inference time 
    # (1.5, 0), # 2x inference time 
]
cache_base_dir="../input_cache"
output_base_dir = "../output/One-to-All-14b"
max_short = 576

MAIN_CHUNK = 81
OVERLAP_FRAMES = 5
FINAL_CHUNK_CANDIDATES = [65, 69, 73, 77, 81]

negative_prompt = [
    "black background, Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
]
black_pose_cfg = True
black_image_cfg = True
controlnet_conditioning_scale = 1.0
case1 = False


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


def build_pipe(device, ckpt_path):
    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=7.0,
        num_train_timesteps=1000,
        use_dynamic_shifting=False
    )

    vae = get_vae('wanx', vae_path, model_dtype)
    encoders = get_text_enc('wanx-t2v', model_path, model_dtype)
    text_encoder = encoders.text_encoder
    tokenizer = encoders.tokenizer

    model = WanTransformer3DModel_Refextractor_2D_Controlnet_prefix.from_config(config_path).to(model_dtype)
    model.set_up_controlnet("configs/wan2.1_t2v_14b_controlnet_1.json", model_dtype)
    model.set_up_refextractor("configs/wan2.1_t2v_14b_refextractor_2d_withmask2.json", model_dtype)
    model.eval()
    model.requires_grad_(False)
    # load ckpt
    checkpoint = {}
    shard_files = [f for f in os.listdir(ckpt_path) if f.endswith(".safetensors")]
    for shard_file in shard_files:
        sd = safe_load(os.path.join(ckpt_path, shard_file), device='cpu')
        checkpoint.update(sd)
    model.load_state_dict(checkpoint, strict=True)

    pipe = WanPipeline(
        transformer=model,
        vae=vae.vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler
    )
    pipe.to(device, dtype=model_dtype)
    return pipe


def build_split_plan(total_len: int):
    ranges = []
    start = 0
    while True:
        current_chunk_end = start + MAIN_CHUNK
        next_chunk_start = start + (MAIN_CHUNK - OVERLAP_FRAMES)
        if next_chunk_start + MAIN_CHUNK >= total_len:
            ranges.append((start, current_chunk_end))
            final_chunk_start = -1
            for length in FINAL_CHUNK_CANDIDATES:
                potential_start = total_len - length
                if potential_start < current_chunk_end - OVERLAP_FRAMES:
                    final_chunk_start = potential_start
                    break
            if final_chunk_start == -1:
                final_chunk_start = next_chunk_start
            ranges.append((final_chunk_start, total_len))
            break
        else:
            ranges.append((start, current_chunk_end))
            start = next_chunk_start
    return ranges


def read_pose_video(pose_mp4_path, transform_fn=None):
    vr = decord.VideoReader(pose_mp4_path)
    fps = vr.get_avg_fps() if vr.get_avg_fps() > 0 else 30
    frames_list = []
    for frame in vr:
        frame_np = frame.asnumpy()
        if transform_fn:
            pil_img = Image.fromarray(frame_np)
            pil_img = transform_fn(pil_img)
            frame_np = np.array(pil_img)
        frames_list.append(frame_np)

    frames_np = np.stack(frames_list)
    tensor = torch.from_numpy(frames_np).float().permute(3, 0, 1, 2) / 255.0 * 2 - 1
    tensor = tensor.unsqueeze(0)
    return tensor, int(fps)


def find_cache_dir(cache_dir_root, ref_path, vid_path, alignmode):
    refname = os.path.splitext(os.path.basename(ref_path))[0]
    vidname = os.path.splitext(os.path.basename(vid_path))[0]
    job_name = f"ref_{refname}_vid_{vidname}"
    pattern = os.path.join(
        cache_dir_root,
        f"ref_{refname}_driven_{vidname}_align_{alignmode}_*"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"❌ 未找到缓存目录: {pattern}")
    if len(matches) > 1:
        print(f"⚠ 匹配到多个缓存目录，取第一个: {matches}")
    return matches[0],job_name


def unified_worker(rank, world_size, task_list, ckpt_path):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    pipe = build_pipe(device, ckpt_path)
    current_date = datetime.now().strftime("%Y-%m-%d")
    



    for idx, (job_name, cache_dir, new_h, new_w, ref_cfg, pose_cfg,prompt) in enumerate(task_list):
        if idx % world_size != rank:
            continue
        print(f"[GPU {rank}] 开始任务: {cache_dir}")
        output_root = os.path.join(output_base_dir, f"{current_date}_ref_{ref_cfg}_pose_{pose_cfg}")
        frames_output_dir = os.path.join(output_root, f"{job_name}_ref_{ref_cfg}_pose_{pose_cfg}")
        os.makedirs(frames_output_dir, exist_ok=True)
        final_video_path = os.path.join(output_root, f"{job_name}_ref_{ref_cfg}_pose_{pose_cfg}.mp4")
        if os.path.exists(final_video_path):
            print(f"[GPU {rank}] {final_video_path} 已存在，跳过")
            continue

        print(f"[GPU {rank}] 处理: {cache_dir}")
        # 读取输入
        image_input = Image.open(os.path.join(cache_dir, "image_input.png")).convert("RGB")
        if new_h and new_w:
            new_h = float(new_h)
            new_w = float(new_w)
        else:
            new_h, new_w = image_input.height, image_input.width

        if min(new_h, new_w) > max_short:
            if new_h < new_w:
                scale = max_short / new_h
                new_h, new_w = max_short, int(new_w * scale)
            else:
                scale = max_short / new_w
                new_w, new_h = max_short, int(new_h * scale)
                
        new_h, new_w = int(new_h//16*16), int(new_w//16*16)
        # new_h, new_w = 1024, 576
        transform_fn = partial(resizecrop, th=new_h, tw=new_w)
        image_input = transform_fn(image_input)

        pose_tensor, pose_fps = read_pose_video(os.path.join(cache_dir, "pose.mp4"),transform_fn)
        pose_input_img = Image.open(os.path.join(cache_dir, "pose_input.png")).convert("RGB")
        pose_input_img = transform_fn(pose_input_img)
        mask_input     = Image.open(os.path.join(cache_dir, "mask_input.png")).convert("L")
        mask_input = transform_fn(mask_input)

        mask_np        = np.array(mask_input, dtype=np.float32) / 255.0
        mask_tensor    = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).unsqueeze(2)

        src_pose_tensor = torch.from_numpy(np.array(pose_input_img)).unsqueeze(0).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1
        src_pose_tensor = src_pose_tensor.unsqueeze(2)

        split_plan = build_split_plan(pose_tensor.shape[2])
        all_generated_frames_np = {}

        for idx, (start, end) in enumerate(split_plan):
            # Debug mode: uncomment break to only infer the first segment
            # if idx > 0:
            #     break
            sub_video = pose_tensor[:, :, start:end]  # [B=1, C, T, H, W]
            prev_frames = None
            if start > 0:
                needed_idx = range(start, start + OVERLAP_FRAMES)
                if all(k in all_generated_frames_np for k in needed_idx):
                    prev_frames = [
                        Image.fromarray(all_generated_frames_np[k]) for k in needed_idx
                    ]

            output_chunk = pipe(
                image=image_input,
                image_mask=mask_tensor,
                control_video=sub_video,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=new_h,
                width=new_w,
                num_frames=end - start,
                image_guidance_scale=ref_cfg,
                pose_guidance_scale=pose_cfg,
                num_inference_steps=30,
                generator=torch.Generator(device=device).manual_seed(42),
                black_image_cfg=black_image_cfg,
                black_pose_cfg=black_pose_cfg,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                return_tensor=True,
                case1=case1,
                token_replace=(prev_frames is not None),
                prev_frames=prev_frames,
                image_pose=src_pose_tensor
            ).frames


            output_chunk = (
                output_chunk[0].detach().cpu() / 2 + 0.5
            ).float().clamp(0, 1).permute(1, 2, 3, 0).numpy()
            output_chunk = (output_chunk * 255).astype("uint8")

            for j in range(end - start):
                gidx = start + j
                all_generated_frames_np[gidx] = output_chunk[j]
                imageio.imwrite(os.path.join(frames_output_dir, f"frame_{gidx:06d}.png"), output_chunk[j])

        alpha = 0.6
        frames_combined = []

        src_uint8 = ((pose_tensor / 2 + 0.5).clamp(0, 1) * 255
                    )[0].byte().permute(1, 2, 3, 0).numpy()     # (T,H,W,C)
        sorted_idx = sorted(all_generated_frames_np.keys())


        for t in sorted_idx:
            src  = src_uint8[t].astype(np.float32)
            pred = all_generated_frames_np[t].astype(np.float32)
            blended = (alpha * src + (1 - alpha) * pred).round().astype(np.uint8)
            concat  = np.concatenate([blended, pred.astype(np.uint8)], axis=1)  # (H,2W,C)
            frames_combined.append(concat)

        imageio.mimwrite(
            final_video_path,
            frames_combined,
            fps=pose_fps,
            quality=5
        )

        print(f"[GPU {rank}] 完成视频: {final_video_path}")
    del pipe
    torch.cuda.empty_cache()


def run_all_tasks(ckpt_path, img_g, pose_g, cache_dir_root="input_cache2"):
    """扫描 cache 目录，跑全部任务"""
    # task_list = sorted(glob.glob(os.path.join(cache_dir_root, "*")))
    # ref, vid , mode, new_h, new_w, ref_cfg, pose_cfg, prompt
    # new_h, new_w, ref_cfg, pose_cfg: ""   means use default 
    # prompt : ""   means empty string
    task_list = [
        #
        ("../examples/img.png","../examples/vid.mp4","ref",840,480,"","",""),   
        # text prompt for generation (especially important for misaligned/pose mode cases
        ("../examples/joker2_resize.png","../examples/douyinvid5_v2.mp4","pose","","","","","a clown in a vibrant red suit dancing joyfully on a street, black shoes, with skyscrapers and neon lights in an urban city background, clear hand"),    
        ("../examples/musk.jpg","../examples/vid2.mp4","ref","","","","",""),       
        ("../examples/maodie.png","../examples/vid2.mp4","ref","","","","",""),     
    ]


    task_list_resolved = []
    for ref_path, vid_path , alignmode, new_h, new_w, ref_cfg, pose_cfg, prompt in task_list:
        if not ref_cfg:  ref_cfg  = img_g
        else:            ref_cfg  = float(ref_cfg)
        if not pose_cfg: pose_cfg = pose_g
        else:            pose_cfg = float(pose_cfg)

        try:
            cache_dir,job_name = find_cache_dir(cache_dir_root, ref_path, vid_path, alignmode)
        except FileNotFoundError as e:
            print(e)
            continue

        task_list_resolved.append((job_name,cache_dir,new_h,new_w,ref_cfg,pose_cfg,prompt))


    world_size = torch.cuda.device_count()
    mp.spawn(
        unified_worker,
        args=(world_size, task_list_resolved, ckpt_path),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    for ckpt in ckpt_paths:
        for img_g, pose_g in default_cfg_combos:
            print(f"\n===== ckpt: {ckpt} | img_g={img_g} | pose_g={pose_g} =====")
            run_all_tasks(ckpt, img_g, pose_g, cache_dir_root=cache_base_dir)

import os
import cv2
import torch
import shutil
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import glob

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
base_temp_dir = "./temp_eval_cartoon_frames"

gt_base_dir = "./cartoon_test_set"

model_paths = {
    "14b": "../output_benchmark/cartoon/One-to-All-14b/2025-11-27_cartoon_ref_2.5_pose_1.5",
    # "1.3b": "xxx"
}

def read_video_frames(video_path):
    if not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def read_model_frames(model_name, video_id):
    matches = glob.glob(os.path.join(model_paths[model_name], f"{video_id}*"))
    matches = [m for m in matches if os.path.isdir(m)]
    if not matches:
        raise ValueError(f"⚠ 没找到前缀 {video_id} 对应的目录")
    else:
        folder = matches[0]
        print(f"✅ 匹配到目录: {folder}")
    
    if not os.path.exists(folder):
        return []
    files = sorted(os.listdir(folder))
    return [Image.open(os.path.join(folder, f)).convert('RGB') for f in files]

video_ids = [f.replace(".mp4", "") for f in os.listdir(gt_base_dir) if f.endswith(".mp4")]

for model_name in model_paths.keys():
    print(f"\n=== 处理模型: {model_name} ===")

    for vid_id in video_ids:
        gt_video_path = os.path.join(gt_base_dir, f"{vid_id}.mp4")
        
        if not os.path.exists(gt_video_path):
            print(f"[{model_name}][{vid_id}] 缺少GT视频，跳过")
            continue

        gt_frames = read_video_frames(gt_video_path)
        
        pred_frames = read_model_frames(model_name, vid_id)

        if len(gt_frames) == 0 or len(pred_frames) == 0:
            print(f"[{model_name}][{vid_id}] 缺少帧数据，跳过")
            continue

        min_len = min(len(gt_frames), len(pred_frames))
        gt_frames = gt_frames[:min_len]
        pred_frames = pred_frames[:min_len]
        size = gt_frames[0].size

        temp_pred_video_dir = os.path.join(base_temp_dir, model_name, "pred", vid_id)
        temp_gt_video_dir   = os.path.join(base_temp_dir, model_name, "gt", vid_id)
        shutil.rmtree(temp_pred_video_dir, ignore_errors=True)
        shutil.rmtree(temp_gt_video_dir, ignore_errors=True)
        os.makedirs(temp_pred_video_dir, exist_ok=True)
        os.makedirs(temp_gt_video_dir, exist_ok=True)

        def save_frame(i, pred_frame, gt_frame):
            try:
                pred_resized = pred_frame.resize(size, Image.BICUBIC)
                gt_resized   = gt_frame.resize(size, Image.BICUBIC)
                pred_resized.save(os.path.join(temp_pred_video_dir, f"{i:04d}.png"))
                gt_resized.save(os.path.join(temp_gt_video_dir, f"{i:04d}.png"))
            except Exception as e:
                print(f"[{model_name}][{vid_id}] 保存第 {i} 帧失败: {e}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(
                executor.map(lambda idx: save_frame(idx, pred_frames[idx], gt_frames[idx]),
                             range(min_len)),
                total=min_len,
                desc=f"[{model_name}][{vid_id}] 保存中"
            ))

    print(f"[{model_name}] 处理完成 ✅")

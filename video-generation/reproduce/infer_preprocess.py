import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_utils import load_poses_whole_video, resizecrop

import cv2
import json
import argparse
import numpy as np
import glob
from PIL import Image
from functools import partial

import decord
from unit_test.io_utils import save_video
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Output directory
cache_base_dir="../input_cache_benchmark"

# ref_path, video_path, interval, align, mode, h, w, face_change, head_change, without_face
# interval: Frame interval (sample every N frames)
# align: Whether to perform retargeting (True/False)
# mode: Alignment mode
#       - "ref": Align to reference image
#       - "pose": Use original driving video
# h: Output height
# w: Output width
# face_change: Whether to fade facial landmarks (True: fade, False: normal color)
# head_change: Whether to fade head landmarks (True: fade, False: normal color)
# without_face: Whether to skip drawing facial landmarks (True: skip, False: draw)
# 

def scan_tiktok_dataset(base_dir):
    tasks = []
    all_items = glob.glob(os.path.join(base_dir, "[0-9]*"))
    video_folders = sorted([item for item in all_items if os.path.isdir(item)])
    
    for vf in video_folders:
        folder_name = os.path.basename(vf)
        ref_path = os.path.join(base_dir, f"{folder_name}_reference.png")
        
        if os.path.exists(ref_path):
            tasks.append((ref_path, vf, 1, False, "ref", "", "", True, False, False))
        else:
            print(f"⚠️ Missing reference: {ref_path}")
    return tasks


def scan_cartoon_dataset(base_dir):
    tasks = []
    ref_dir = os.path.join(base_dir, "reference")
    video_files = sorted(glob.glob(os.path.join(base_dir, "*.mp4")))
    
    for vf in video_files:
        vf_name = os.path.splitext(os.path.basename(vf))[0]
        ref_path = os.path.join(ref_dir, f"{vf_name}.png")
        
        if os.path.exists(ref_path):
            tasks.append((ref_path, vf, 1, False, "ref", "", "", True, False, False))
        else:
            print(f"⚠️ Missing reference: {ref_path}")
    return tasks

task_list = []
tiktok_test_set = "../benchmark/tiktok_test_set/"
task_list.extend(scan_tiktok_dataset(tiktok_test_set))

cartoon_test_set = "../benchmark/cartoon_test_set/"
task_list.extend(scan_cartoon_dataset(cartoon_test_set))

print(f"总共找到 {len(task_list)} 个任务")


def process_one(reference_path, video_path, frame_interval, do_align, alignmode, h=None, w=None, face_change=True, head_change=True, without_face=False):

    if not h or not w:
        ref_img_tmp = Image.open(reference_path).convert("RGB")
        w, h = ref_img_tmp.size
    else:
        h = int(h)
        w = int(w)


    max_short = 768
    if min(h, w) > max_short:
        if h < w:
            scale = max_short / h
            h, w = max_short, int(w * scale)
        else:
            scale = max_short / w
            w, h = max_short, int(h * scale)
    new_h = (h // 16) * 16
    new_w = (w // 16) * 16
    transform = partial(resizecrop, th=new_h, tw=new_w)
    anchor_idx = 0

    pose_tensor, image_input, pose_input, mask_input = load_poses_whole_video(
        video_path=video_path,
        reference=reference_path,
        frame_interval=frame_interval,
        do_align=do_align,
        transform=transform,
        alignmode=alignmode,
        anchor_idx=anchor_idx,
        face_change=face_change,
        head_change=head_change,
        without_face=without_face,
    )


    if os.path.isdir(video_path):
        fps = 30
        print(f"⚠ {video_path} isdir, use default fps={fps}")
    else:
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()

    output_fps = fps / frame_interval

 
    video_tensor = pose_tensor / 255 * 2 - 1


    vidname = os.path.splitext(os.path.basename(video_path))[0]
    refname = os.path.splitext(os.path.basename(reference_path))[0]
    cache_dir = os.path.join(cache_base_dir, f"ref_{refname}_driven_{vidname}_align_{alignmode}_h_{new_h}_w_{new_w}_fps_{output_fps}")
    os.makedirs(cache_dir, exist_ok=True)


    save_video(os.path.join(cache_dir, "pose.mp4"), video_tensor, fps=output_fps)
    image_input.save(os.path.join(cache_dir, "image_input.png"))
    pose_input.save(os.path.join(cache_dir, "pose_input.png"))
    mask_input.save(os.path.join(cache_dir, "mask_input.png"))

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for ref_path, video_path , interval, align, mode, h, w, face_change, head_change, without_face in task_list:
            futures.append(executor.submit(process_one, ref_path, video_path, interval, align, mode, h, w, face_change, head_change, without_face))
        
        for f in tqdm(futures):
            f.result()



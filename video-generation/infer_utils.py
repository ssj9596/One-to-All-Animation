from opensora.dataset.utils import draw_pose_aligned
import os
import torch
from einops import rearrange
import copy
import random
import math
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import decord
from functools import partial
import glob
from dwpose_utils.dwpose_detector import dwpose_detector_aligned
from wanpose_utils.pose2d import Pose2d

from infer_function import *

pose2d_checkpoint_path = "../pretrained_models/process_checkpoint/pose2d/vitpose_h_wholebody.onnx"
det_checkpoint_path = "../pretrained_models/process_checkpoint/det/yolov10m.onnx"

wanpose2d = Pose2d(checkpoint=pose2d_checkpoint_path, detector_checkpoint=det_checkpoint_path)


def warp_ref_to_pose(tgt_img,
                     ref_pose: dict, #driven pose 
                     tgt_pose: dict,
                     bg_val=(0, 0, 0),
                     conf_th=0.9,
                     align_center=False):

    H, W = tgt_img.shape[:2]
    img_tgt_pose = draw_pose_aligned(tgt_pose, H, W, without_face=True)

    tgt_kpt = tgt_pose['bodies']['candidate'].astype(np.float32)
    ref_kpt = ref_pose['bodies']['candidate'].astype(np.float32)

    scale_ratio = scale_and_translate_pose(tgt_pose, ref_pose, conf_th=conf_th, return_ratio=True)

    anchor_idx = 1
    x0 = tgt_kpt[anchor_idx][0] * W
    y0 = tgt_kpt[anchor_idx][1] * H

    ref_x = ref_kpt[anchor_idx][0] * W if not align_center else W/2
    ref_y = ref_kpt[anchor_idx][1] * H

    dx = ref_x - x0
    dy = ref_y - y0

    # 仿射变换矩阵
    M = np.array([[scale_ratio, 0, (1-scale_ratio)*x0 + dx],
                  [0, scale_ratio, (1-scale_ratio)*y0 + dy]],
                 dtype=np.float32)
    img_warp = cv2.warpAffine(tgt_img, M, (W, H),
                              flags=cv2.INTER_LINEAR,
                              borderValue=bg_val)
    img_tgt_pose_warp = cv2.warpAffine(img_tgt_pose, M, (W, H),
                                       flags=cv2.INTER_LINEAR,
                                       borderValue=bg_val)
    zeros = np.zeros((H, W), dtype=np.uint8)
    mask_warp = cv2.warpAffine(zeros, M, (W, H),
                               flags=cv2.INTER_NEAREST,
                               borderValue=255)
    return img_warp, img_tgt_pose_warp, mask_warp




def load_poses_whole_video(video_path,
                           reference,
                           frame_interval= 2,
                           transform     = None,
                           do_align      = False,
                           alignmode     = "ref",  #align to reference or align to pose
                           face_change   = False,  #lighten color
                           head_change   = False,  #lighten color
                           anchor_idx    = 0,
                           without_face  = False ):

    # ---------------- ref dwpose ---------------- #
    ref_pose      = None
    ref_pose_tensor = None
 
    ref_img = Image.open(reference).convert('RGB')

    h_ref, w_ref = ref_img.height, ref_img.width
    ref_rgb = np.array(ref_img)

    ref_pose_meta = wanpose2d([ref_rgb])[0]
    ref_dwpose = aaposemeta_to_dwpose(ref_pose_meta)

    # ---------------- driven pose ---------------- #
    if os.path.isdir(video_path):
        img_paths = sorted(glob.glob(os.path.join(video_path, "*.png")))
        assert img_paths, f"NO PNG in {video_path}!!"
        frames_list = []
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            img = resizecrop(img, th=h_ref, tw=w_ref)
            frames_list.append(np.array(img))
        frames = np.stack(frames_list)
        total_frames = len(frames_list)
        frame_indices = np.arange(0, total_frames, frame_interval).astype(int)
        frames = frames[frame_indices]
        h, w = frames[0].shape[:2]
    else:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        frame_indices = np.arange(0, total_frames, frame_interval).astype(int)
        frames = vr.get_batch(frame_indices).asnumpy()
        frames_resized = []
        for fr in frames:
            pil_fr = Image.fromarray(fr)
            pil_fr = resizecrop(pil_fr, th=h_ref, tw=w_ref)
            frames_resized.append(np.array(pil_fr))
        frames = np.stack(frames_resized)
        h, w = frames[0].shape[:2]

    tpl_pose_metas = wanpose2d(frames)
    tpl_dwposes = [aaposemeta_to_dwpose(meta) for meta in tpl_pose_metas]

    # ---------------- retarget ---------------- #

    if alignmode == "ref":
        image_input = transform(ref_img)
        pose_input =  draw_pose_aligned(ref_dwpose, h, w, without_face=True)
        pose_input = transform(Image.fromarray(pose_input))
        mask_input = Image.new("RGB", image_input.size, (0, 0, 0))
        if do_align:
            tpl_dwposes = align_to_reference(ref_pose_meta, tpl_pose_metas, tpl_dwposes, anchor_idx)



    elif alignmode == "pose":
        # import pdb;pdb.set_trace()
        image_input, pose_input, mask_input = warp_ref_to_pose(ref_rgb, tpl_dwposes[anchor_idx], ref_dwpose)
        image_input = transform(Image.fromarray(image_input))
        pose_input = transform(Image.fromarray(pose_input))
        mask_input = transform(Image.fromarray(mask_input).convert("RGB"))
        # rescale-ref and change part of pose
        if do_align:
            tpl_dwposes = align_to_pose(ref_dwpose, tpl_dwposes, anchor_idx)

    # ---------------- draw pose & transform ---------------- #
    pose_imgs = []
    for pose_np in tpl_dwposes:
        pose_img = draw_pose_aligned(pose_np, h, w, without_face=without_face,face_change=face_change,head_change=head_change)
        pose_img = transform(Image.fromarray(pose_img))
        pose_img = torch.from_numpy(np.array(pose_img))
        pose_img = rearrange(pose_img, 'h w c -> c h w')
        pose_imgs.append(pose_img)

    pose_tensor = torch.stack(pose_imgs)      # (T,C,H,W)

    return pose_tensor, image_input, pose_input, mask_input


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




if __name__ == "__main__":
    video_path = "xxx"
    ref_image_path = "xxx"
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    new_height = 1024
    new_width = 572
    transform = partial(resizecrop, th=new_height, tw=new_width)
    print("video_fps:",fps)

    # 调用函数
    pose_tensor,image_input, pose_input, mask_input = load_poses_whole_video(
        video_path=video_path,
        reference=ref_image_path,  
        frame_interval=1,
        transform=transform,  
        do_align = True,
        alignmode="pose",   
        face_change=True,
        head_change=False,
        without_face=False,
        anchor_idx=0,
    )
    import pdb;pdb.set_trace()
    from unit_test.io_utils import save_video
    video_tensor = pose_tensor / 255 * 2 - 1
    save_video("pose.mp4", video_tensor,fps=fps)

    # print("Pose tensor shape:", pose_tensor.shape)


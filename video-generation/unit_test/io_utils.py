import os
import torch
import numpy as np
from torchvision.utils import save_image
import imageio
from einops import rearrange

def save_images(save_img_root, img_tensors):
    '''
    tensors are normalized into (-1, 1)
    '''
    os.makedirs(save_img_root, exist_ok=True)
    for i in range(img_tensors.shape[0]):
        save_image(img_tensors[i], f"{save_img_root}/{str(i).zfill(8)}.png", 
        normalize=True, value_range=(-1, 1))

def save_video(save_video_path, img_tensors,fps=16):
    # os.makedirs(os.path.basename(save_video_path), exist_ok=True)
    img_tensors = (img_tensors / 2 + 0.5).clamp(0, 1) * 255
    img_tensors = rearrange(img_tensors, 't c h w -> t h w c').contiguous()
    img_tensors = img_tensors.cpu().numpy().astype(np.uint8)
    imageio.mimwrite(save_video_path, img_tensors, fps=fps, quality=4, output_params=["-loglevel", "error"]) # Highest quality is 10, lowest is 0
import time
import traceback
import pandas as pd

try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
import glob
import json
import os, io, csv, math, random
import cv2 as cv
import numpy as np
import torchvision
from einops import rearrange
from decord import VideoReader
from os.path import join as opj
from collections import Counter
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid

import torch
import torchvision.transforms as transforms 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm
from PIL import Image
from accelerate.logging import get_logger

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing
from opensora.models.causalvideovae import ae_norm
from .transform import get_video_transform, get_mask_transform
from torchvision.transforms import Lambda
import tarfile
import multiprocessing
import sys
from .utils import *
# fix numpy bug
sys.modules.setdefault('numpy._core', np.core)
if 'numpy._core.multiarray' not in sys.modules:
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
if 'numpy._core._multiarray_umath' not in sys.modules:
    sys.modules['numpy._core._multiarray_umath'] = np.core.multiarray


logger = get_logger(__name__)
# multiprocessing.set_start_method('spawn')
def is_video_file(path):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.mpeg', '.mpg']
    ext = os.path.splitext(path)[1]
    return ext.lower() in video_extensions

def filter_json_by_existed_files(directory, data, postfixes=[".mp4", ".jpg"]):
    # 构建搜索模式，以匹配指定后缀的文件
    matching_files = []
    for postfix in postfixes:
        pattern = os.path.join(directory, '**', f'*{postfix}')
        matching_files.extend(glob.glob(pattern, recursive=True))

    # 使用文件的绝对路径构建集合
    mp4_files_set = set(os.path.abspath(path) for path in matching_files)

    # 过滤数据条目，只保留路径在mp4文件集合中的条目
    filtered_items = [item for item in data if item['path'] in mp4_files_set]

    return filtered_items

def to_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str) and x.strip():
        try:
            return list(ast.literal_eval(x))
        except Exception:
            pass
    return []

def random_video_noise(t, c, h, w):
    vid = torch.rand(t, c, h, w) * 255.0
    vid = vid.to(torch.uint8)
    return vid

def prompt_cleaning(cap):
    assert isinstance(cap, str)
    return cap.replace('\n', ' ').replace('\t', ' ')

def crop_frame(frame, target_w, target_h):
    w, h = frame.size
    crop_x = w // 2 - target_w // 2
    crop_y = h // 2 - target_h // 2
    cropped_frame = frame.crop((crop_x, crop_y, crop_x + target_w, crop_y+target_h))
    resized_frame = cropped_frame.resize((w, h), Image.BICUBIC)
    return resized_frame

def zoomed_in(frame, zoom_factor=0.2, num_frames=49):
    width, height = frame.size
    smallest_w, smallest_h = int(width * (1-zoom_factor)), int(height * (1-zoom_factor))
    width_per_frame = np.linspace(width, smallest_w, num_frames)
    height_per_frame = np.linspace(height, smallest_h, num_frames)
    result_frames = [np.array(frame)]
    for i in range(1, num_frames):
        frame_i = crop_frame(frame, width_per_frame[i], height_per_frame[i])
        result_frames.append(np.array(frame_i))
    result_frames = np.stack(result_frames)
    return result_frames

def get_embd_condition(pred, visiable, d, sample_type, height, width, points_num = None):
    '''
    trajectory: B F N 2
    contidion: B F N
    sample_type: random, grid, sparse
    '''
    batch, frames, L, _ = pred.shape
    if sample_type == "random":
        points_num = 0.9 * L # train stage 1 dense sample 
        points_index = random.sample(list(range(L)), points_num)
        pred = pred[:, :, points_index, :]
        visiable = visiable[:, :, points_index, :]
        pos_embedding = get_1d_sincos_pos_embed_from_grid(d, torch.tensor(list(range(points_num))))
        indices = torch.randperm(points_num)
        pos_embedding = pos_embedding[indices]
        condition = np.zeros((batch, d, frames, height, width))
        for t in range(frames):
            for b in range(batch):
                for n in range(len(pos_embedding)):
                    x_n = int(pred[b, t, n, 0]) # w
                    y_n = int(pred[b, t, n, 1]) # h
                    if x_n >= width or x_n < 0:
                        continue
                    if y_n >= height or y_n < 0:
                        continue
                    condition[b, :, t, y_n, x_n] = visiable[b, t, n].item() * pos_embedding[n,:]
        return condition
    elif sample_type == "grid":
        pass
        return None
    elif sample_type == "sparse":
        points_num = 0.1 * L # train stage 2 sparse sample 
        points_index = random.sample(list(range(L)), points_num)
        pred = pred[:, :, points_index, :]
        visiable = visiable[:, :, points_index, :]
        pos_embedding = get_1d_sincos_pos_embed_from_grid(d, torch.tensor(list(range(points_num))), output_type="pt")
        indices = torch.randperm(points_num)
        pos_embedding = pos_embedding[indices]
        condition = np.zeros((batch, d, frames, height, width))
        for t in range(frames):
            for b in range(batch):
                for n in range(len(pos_embedding)):
                    x_n = int(pred[b, t, n, 0]) # w
                    y_n = int(pred[b, t, n, 1]) # h
                    if x_n >= width or x_n < 0:
                        continue
                    if y_n >= height or y_n < 0:
                        continue
                    condition[b, :, t, y_n, x_n] = visiable[b, t, n].item() * pos_embedding[n,:]
        return condition
    return None

class BodyDance_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer, transform_topcrop, temporal_downscale, device='cuda', one_more_second=False, ignore_timestamps=False, valid=False, load_cached_latents=False, null_embedding_path=None):
        self.data = args.data

        self.train_fps = args.train_fps
        self.transform = transform
        self.transform_topcrop = transform_topcrop
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.cfg = args.cfg
        self.one_more_second=one_more_second
        self.without_face = 0.2
        self.pose_cfg = args.pose_cfg
        self.refimg_cfg = args.refimg_cfg
        self.hand_mask = args.hand_mask
        self.refimg_crop = args.refimg_crop
        self.video_crop = args.zoomin
        self.only_ref  = args.only_ref
        self.i2v = args.i2v
        print("\n=== 参数配置 ===")
        print(f"self.pose_cfg: {self.pose_cfg:.2f}")
        print(f"self.refimg_cfg: {self.refimg_cfg:.2f}")
        print(f"self.hand_mask: {self.hand_mask}")
        print(f"self.refimg_crop: {self.refimg_crop:.2f}")
        print(f"self.video_crop: {self.video_crop:.2f}")
        print(f"self.only_ref: {args.only_ref}")
        print(f"self.i2v: {args.i2v}")

        self.drop_score_ratio = 0.2
        self.load_cached_latents = load_cached_latents
        self.args = args
        # self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device).eval()
        # self.model.requires_grad_(False)
        if self.load_cached_latents:
            self.null_text_embedding = torch.load(null_embedding_path, map_location='cpu', weights_only=True)
        self.temporal_downscale = temporal_downscale
        self.valid = valid
        self.simulate = True
        
        if valid:
            self.drop_score_ratio = 0
            self.cfg = 0
        self.v_decoder = DecordInit()
        self.support_Chinese = False
        if not ('mt5' in args.text_encoder_name):
            self.support_Chinese = False
        if isinstance(self.data, list) and len(self.data) == 1:
            self.data = self.data[0]
        if isinstance(self.data, str) and self.data.endswith('txt'):
            self.data = [line.strip() for line in open(self.data).readlines()]
        if isinstance(self.data, list):
            df = pd.DataFrame()
            for csv_file in self.data:
                df_ = pd.read_csv(csv_file)
                df = pd.concat([df, df_], ignore_index=True)
            self.data = df
        else:
            self.data = pd.read_csv(self.data)

        if one_more_second and 'end_frame' in self.data:
            self.data['end_frame'] = self.data['end_frame'] + self.data['fps'] * 2 #会遇到多重的取整损失，
            self.data['end_frame'] = self.data[['end_frame', 'num_frames']].min(axis=1)

        if ignore_timestamps and 'end_frame' in self.data:
            # self.data = self.data.drop(['start_frame', 'end_frame'], axis=1)
            self.data['end_frame'] = self.data['num_frames'] #cancel out end_frame的限制

        self.data['id'] = np.arange(len(self.data))
        self.norm_fun = Lambda(lambda x: 2. * x - 1.)

        self.verbose = False

    def set_bucket_info(self, bucket_info):
        self.bucket_info = bucket_info

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            if not self.verbose:
                print(idx)
                self.verbose = True
            data = self.get_data(idx)
            return data
        except Exception as e:
            print(e)
            print('get data error at idx:', idx)
            index, num_frames, height, width = [int(val) for val in idx.split("-")]
            path = self.data.iloc[index]['path']
            print('get data error at path:', path)

            new_idx = f"{index+1}-{num_frames}-{height}-{width}"
            new_data = self.get_data(new_idx)
            return new_data

            raise Exception(e)

    def get_data(self, idx):
        index, num_frames, height, width = [int(val) for val in idx.split("-")]
        path = self.data.iloc[index]['path']
        # return self.get_video(idx)
        if is_video_file(path):
            return self.get_video(idx)
        else:
            return self.get_image(idx)
    
    def get_video(self, idx):
        # import pdb; pdb.set_trace()
        idx, num_frames, height, width = [int(val) for val in idx.split("-")]
        real_t, real_h, real_w = num_frames, height, width

        video, pose, reference_frame, face_mask, ocr_mask, hand_mask, ref_face_mask, reference_pose_tensor = self.read_frames(idx, real_t)

        vid_face_bbox = get_bbox_from_mask(face_mask)
        ref_face_bbox = get_bbox_from_mask(ref_face_mask)


        video, pose, reference_frame, face_mask, ocr_mask, hand_mask, reference_pose_tensor, inpaint_mask = apply_random_cropping_strategy_v2(
            video, pose, reference_frame, face_mask, ocr_mask, hand_mask, reference_pose_tensor,
            vid_face_bbox, ref_face_bbox,
            real_h, real_w,
            pose_cfg=self.pose_cfg, 
            refimg_cfg = self.refimg_cfg,
            refimg_crop=self.refimg_crop, 
        )

        face_mask = torch.maximum(face_mask, hand_mask)

        transform = get_video_transform(real_h, real_w, self.norm_fun)
        # 只做resize, 不需要归一化
        mask_transform = get_mask_transform(real_h, real_w)

        video = transform(video)  # T C H W -> T C H W
        pose = transform(pose)
        reference_frame = transform(reference_frame)
        reference_pose_tensor = transform(reference_pose_tensor)

        face_mask = mask_transform(face_mask)
        ocr_mask = mask_transform(ocr_mask)
        inpaint_mask = mask_transform(inpaint_mask)

        video = video.transpose(0, 1)  # T C H W -> C T H W
        pose = pose.transpose(0, 1)
        
        reference_frame = reference_frame.transpose(0, 1)
        reference_pose_tensor = reference_pose_tensor.transpose(0, 1)
        face_mask = face_mask.transpose(0, 1)
        ocr_mask = ocr_mask.transpose(0, 1)
        inpaint_mask = inpaint_mask.transpose(0, 1)

        reference_frame = torch.cat([reference_frame,inpaint_mask],dim=0)


        # text = row['qwen_caption']
        text = self.data.iloc[idx]['qwen_caption']
        text = str(text) if pd.notna(text) else ''
        if not isinstance(text, list):
            text = [text]
        text = prompt_cleaning(random.choice(text))
        text = text
        if random.random() > self.cfg:
            text = text_preprocessing(text, support_Chinese=self.support_Chinese) 
        else:
            cfg_embed_scale = 6
            text = ""
        cfg_embed_scale = 6

        condition = pose

        return dict(pixel_values=video, caption=text, cfg_embed_scale = cfg_embed_scale, 
                        condition = condition, reference_frame = reference_frame, face_mask = face_mask, ocr_mask = ocr_mask)

    def get_image(self, idx):
        # TODO: 对于图像数据，暂时没有vae缓存的逻辑
        idx, num_frames, height, width = [int(val) for val in idx.split("-")]
        real_t, real_h, real_w = num_frames, height, width

        row = self.data.iloc[idx]
        ori_height = row['height']
        ori_width = row['width']   
        img_pairs = row['img_pairs']
        if isinstance(img_pairs, str):        
            img_pairs = eval(img_pairs)
        
        pair = random.choice(img_pairs)
        src_path      = pair['src_path'] 
        tgt_path      = pair['tgt_path'] 
        tgt_pose_path = pair['tgt_pose_path']
        text = pair['qwen_caption']

        src_pose_path = os.path.splitext(src_path)[0] + "_pose.npy"

        if self.only_ref:
            tgt_path = src_path
            tgt_pose_path = src_pose_path

        is_same_image = (src_path == tgt_path)

    
        # 1) 图像 → Tensor (1, 3, H, W)
        def _read_rgb(path):
            img = Image.open(path).convert("RGB")
            img = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)  # (3,H,W)
            return img.unsqueeze(0)   # (1,3,H,W)

        reference_frame = _read_rgb(src_path)
        image    = _read_rgb(tgt_path)

        pose_data = np.load(tgt_pose_path, allow_pickle=True) 
        ref_pose_data = np.load(src_pose_path, allow_pickle=True) 

        def has_cartoon_keyword(text):
            if isinstance(text, list):
                text = " ".join(text)
            text = str(text)
            kws = ("cartoon", "animate", "anime")
            return any(kw in text.lower() for kw in kws)

        without_face  = has_cartoon_keyword(text) or random.random() < self.without_face
        pose = draw_pose_aligned(pose_data[0], ori_height, ori_width, without_face=without_face)
        ref_pose = draw_pose_aligned(ref_pose_data[0], ori_height, ori_width, without_face=True)


        pose_tensor = torch.from_numpy(np.array([pose])).permute(0, 3, 1, 2)
        ref_pose_tensor = torch.from_numpy(np.array([ref_pose])).permute(0, 3, 1, 2)

        face_mask = draw_face_mask_frompose(pose_data, [0], 1, ori_height, ori_width, for_vae = True).unsqueeze(1)
        ocr_mask = torch.ones((1, 1, real_h, real_w), dtype=torch.float32)

        image, pose_tensor, reference_frame, ref_pose_tensor, face_mask, inpaint_mask =  apply_image_augmentation_strategy_v2(
            image, pose_tensor, reference_frame, ref_pose_tensor, face_mask,
            is_same_image,
            real_h, real_w,
            pose_cfg=self.pose_cfg,
            refimg_cfg = self.refimg_cfg,
            refimg_crop = self.refimg_crop,
        )



        transform = get_video_transform(real_h, real_w, self.norm_fun)
        mask_transform = get_mask_transform(real_h, real_w)

        image = transform(image)
        pose_tensor = transform(pose_tensor)
        
        reference_frame = transform(reference_frame)
        ref_pose_tensor = transform(ref_pose_tensor)

        image = image.transpose(0, 1) 
        pose_tensor = pose_tensor.transpose(0, 1) 
        reference_frame = reference_frame.transpose(0, 1) 
        ref_pose_tensor = ref_pose_tensor.transpose(0, 1) 
        face_mask = mask_transform(face_mask).transpose(0, 1)
        ocr_mask = mask_transform(ocr_mask).transpose(0, 1)
        inpaint_mask = mask_transform(inpaint_mask).transpose(0, 1)

        # CTHW
        reference_frame = torch.cat([reference_frame,inpaint_mask],dim=0)

        text = str(text) if pd.notna(text) else ''
        if not isinstance(text, list):
            text = [text]
        text = prompt_cleaning(random.choice(text))
        text = text
        if random.random() > self.cfg:
            text = text_preprocessing(text, support_Chinese=self.support_Chinese) 
        else:
            cfg_embed_scale = 6
            text = ""
        cfg_embed_scale = 6

        condition = pose_tensor
        return dict(pixel_values=image, caption=text, cfg_embed_scale = cfg_embed_scale, 
                        condition = condition, reference_frame = reference_frame, face_mask = face_mask, ocr_mask = ocr_mask)

    def read_frames(self, idx, real_t):
        row = self.data.iloc[idx]
        decord_vr = self.v_decoder(row['path'])
        pose_data = np.load(row['pose_path'],allow_pickle=True)
        ori_height = row['height']
        ori_width = row['width']    

        start_frame_idx =  row['start_frame']    
        end_frame_idx = row['end_frame']  
        all_ocr_bbox = row['all_ocr_bbox']

        vid_fps = np.round(row['fps'])

        train_fps = self.train_fps 
        if vid_fps >= 50:
            train_fps = self.train_fps if  random.random() < 0.5 else 2 *  self.train_fps
            
        frame_interval = max(int(vid_fps / self.train_fps),1)
        # -------- ① 处理 train_start_index --------
        start_list = to_list(row.get('train_start_index', None))
        if len(start_list) > 3:                       # 去掉首尾
            start_list = start_list[1:-1]
        if start_list:                               # 按预定义起点
            start_idx = random.choice(start_list)
            train_indices = np.arange(start_idx, start_idx + real_t * frame_interval, frame_interval).astype(int)
        else:                                        # 原逻辑
            frame_indices = np.arange(start_frame_idx+1, end_frame_idx-1, frame_interval).astype(int)
            frame_indices = frame_indices[frame_indices < end_frame_idx]
            valid_frame_count = ((len(frame_indices) - 1) // 4) * 4 + 1
            train_indices = frame_indices[:valid_frame_count]
            if len(train_indices) > real_t:
                rs = random.randint(0, len(train_indices) - real_t)
                train_indices = train_indices[rs:rs + real_t]
        real_t = len(train_indices) 

         # -------- ref_index：优先用 refimg_index，排除首尾 --------
        ref_list = to_list(row.get('refimg_index', None))
        if not ref_list:
            ref_list = np.arange(1, len(decord_vr) - 1)
        ref_index = select_refimg_index(ref_list,train_indices)
        if random.random() < self.i2v:
            ref_index = train_indices[0]
            
        video = decord_vr.get_batch(train_indices).asnumpy()

        without_face = random.random() < self.without_face
        pose = get_batch_pose(pose_data, train_indices, ori_height, ori_width,without_face)
        reference_frame = decord_vr[ref_index].asnumpy()
        reference_pose = draw_pose_aligned(pose_data[ref_index], ori_height, ori_width, without_face=True)
        reference_pose_tensor = torch.from_numpy(np.array([reference_pose])).permute(0, 3, 1, 2)

        image_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
        pose_tensor = torch.from_numpy(pose).permute(0, 3, 1, 2)
        reference_frame = torch.from_numpy(reference_frame).unsqueeze(0).permute(0, 3, 1, 2)

        try:
            all_ocr_bbox = eval(str(all_ocr_bbox)) if all_ocr_bbox and not pd.isna(all_ocr_bbox) else []
        except:
            all_ocr_bbox = []
        ocr_mask = draw_ocr_mask(all_ocr_bbox, train_indices, real_t, ori_height, ori_width, for_vae = True).unsqueeze(1)
        

        face_mask = draw_face_mask_frompose(pose_data, train_indices, real_t, ori_height, ori_width, for_vae = True).unsqueeze(1)
        ref_face_mask = draw_face_mask_frompose(pose_data, [ref_index], 1, ori_height, ori_width, for_vae = True).unsqueeze(1)

        hand_mask = torch.zeros_like(face_mask)
        if self.hand_mask:
            hand_mask = draw_hand_mask(pose_data, train_indices, real_t, ori_height, ori_width, for_vae = True).unsqueeze(1)

        return image_tensor, pose_tensor, reference_frame, face_mask, ocr_mask, hand_mask, ref_face_mask, reference_pose_tensor


    def decord_read(self, path, predefine_num_frames, start_frame_idx = 0, total_frames = None):
        decord_vr = self.v_decoder(path)
        if total_frames is None:
            total_frames = len(decord_vr)
        fps = decord_vr.get_avg_fps() if decord_vr.get_avg_fps() > 0 else 30.0
        frame_interval = int(np.round(fps / self.train_fps))
        if frame_interval == 0:
            frame_interval = 1
        frame_indices = np.arange(start_frame_idx, total_frames, frame_interval).astype(int) #total_frames不是整数的时候会取整
        frame_indices = frame_indices[frame_indices < total_frames]
        frames_count = len(frame_indices)
        valid_frame_count = ((frames_count - 1) // 8) * 8 + 1
        frame_indices = frame_indices[:valid_frame_count]
        if len(frame_indices) > predefine_num_frames:
            # if not self.valid:
            #     begin_index, end_index = self.temporal_sample(len(frame_indices), predefine_num_frames)
            #     frame_indices = frame_indices[begin_index: end_index]
            # else:
            frame_indices = frame_indices[:predefine_num_frames] #不要这个随机性了
        video_data = decord_vr.get_batch(frame_indices).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data
    
    
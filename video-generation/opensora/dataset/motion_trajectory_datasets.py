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
from .transform import get_video_transform
from torchvision.transforms import Lambda
import tarfile
logger = get_logger(__name__)

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

class T2VMultiRes_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer, transform_topcrop, temporal_downscale, one_more_second=False, ignore_timestamps=False, valid=False, load_cached_latents=False, null_embedding_path=None):
        self.data = args.data
        self.num_frames = args.num_frames
        self.train_fps = args.train_fps
        self.use_image_num = args.use_image_num
        self.use_img_from_vid = args.use_img_from_vid
        self.transform = transform
        self.transform_topcrop = transform_topcrop
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.cfg = args.cfg
        self.one_more_second=one_more_second
        self.drop_score_ratio = 0.2
        self.load_cached_latents = load_cached_latents
        self.args = args
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
            raise Exception(e)

    def get_data(self, idx):
        index, num_frames, height, width = [int(val) for val in idx.split("-")]
        path = self.data.iloc[index]['path']
        if is_video_file(path):
            return self.get_video(idx)
        else:
            return self.get_image(idx)
    
    def get_video(self, idx):
        idx, num_frames, height, width = [int(val) for val in idx.split("-")]
        video_path = self.data.iloc[idx]['path']
        if 'start_frame' in self.data.iloc[idx] and 'end_frame' in self.data.iloc[idx] and not pd.isna(self.data.iloc[idx]['start_frame']) and not pd.isna(self.data.iloc[idx]['end_frame']):
            start_frame_idx = self.data.iloc[idx]['start_frame']
            end_frame_idx = self.data.iloc[idx]['end_frame']
        else:
            start_frame_idx = 0
            end_frame_idx = None
        real_t, real_h, real_w = num_frames, height, width
        # condition,trajectory
        condition = None
        if self.args.load_condition:
            trajectory = os.path.splitext(video_path)[0]+f'_{start_frame_idx}_{end_frame_idx}_pred_{real_h}_{real_w}.pt'
            condition = os.path.splitext(video_path)[0]+f'_{start_frame_idx}_{end_frame_idx}_visi_{real_h}_{real_w}.pt'
            # assert os.path.exists(trajectory), f"file {trajectory} do not exist!"
            if os.path.exists(trajectory) and os.path.exists(condition):
                trajectory = torch.load(trajectory, weights_only=True)
                condition = torch.load(condition, weights_only=True)
                condition = get_embd_condition(trajectory, condition, dim = self.args.condition_dim, sample_type = self.args.sample_type, height = real_h, width = real_w)
            else:
                trajectory = None
                condition = None
        # assert os.path.exists(video_path), f"file {video_path} do not exist!"
        vae_embedding_path = os.path.splitext(video_path)[0]+'%06d_vae_hunyuan_512.pt'%(start_frame_idx)
        text_embedding_path = os.path.splitext(video_path)[0]+'%06d_text_hunyuan_512.pt'%(start_frame_idx)
        cfg_embed_scale = 6
        if self.load_cached_latents:
            num_latents = (real_t - 1) // self.temporal_downscale + 1
            video = torch.load(vae_embedding_path, map_location='cpu', weights_only=True).squeeze(0)[:, :num_latents]
            text = torch.load(text_embedding_path, map_location='cpu', weights_only=True) #注意是一个字典
            if random.random() < self.cfg:
                text = self.null_text_embedding
                cfg_embed_scale = 1
        else:
            video = self.decord_read(video_path, predefine_num_frames=real_t, start_frame_idx = start_frame_idx, total_frames = end_frame_idx)
            transform = get_video_transform(real_h, real_w, self.norm_fun)
            video = transform(video)  # T C H W -> T C H W
            video = video.transpose(0, 1)  # T C H W -> C T H W
            text = self.data.iloc[idx]['text']
            if not isinstance(text, list):
                text = [text]
            text = prompt_cleaning(random.choice(text))
            text = text
            if random.random() > self.cfg:
                text = text_preprocessing(text, support_Chinese=self.support_Chinese) 
            else:
                cfg_embed_scale = 1
                text = ""
        return dict(pixel_values=video, caption=text, cfg_embed_scale = cfg_embed_scale, condition = condition)

    def get_image(self, idx):
        # TODO: 对于图像数据，暂时没有vae缓存的逻辑
        idx, num_frames, height, width = [int(val) for val in idx.split("-")]
        image_path = self.data.iloc[idx]['path']
        real_t, real_h, real_w = num_frames, height, width
        if os.path.isfile(image_path):
            try:
                image = Image.open(image_path).convert('RGB')  # [h, w, c]
            except:
                image = cv.imread(image)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = Image.fromarray(image)
        else:
            tarname = os.path.dirname(image_path)
            basename = os.path.basename(image_path)
            assert tarname.endswith('.tar')
            with tarfile.open(tarname, 'r') as tar:
                with tar.extractfile(basename) as image_data:
                    image = np.frombuffer(image_data.read(), np.uint8)
                    image = cv.imdecode(image, cv.IMREAD_UNCHANGED).astype(np.uint8)
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
        #NOTE: only for i2v testing
        if self.simulate:
            image = torch.from_numpy(zoomed_in(image))
            image = rearrange(image, 't h w c ->t c h w')
        else:
            image = torch.from_numpy(np.array(image))  # [h, w, c]
            image = rearrange(image, 'h w c -> c h w').unsqueeze(0)  #  [1 c h w]
        transform = get_video_transform(real_h, real_w, self.norm_fun)
        image = transform(image)
        image = image.transpose(0, 1) 

        #TODO: 考虑实现multiple caption
        text = self.data.iloc[idx]['text']
        if not isinstance(text, list):
            text = [text]
        text = prompt_cleaning(random.choice(text))
        if self.simulate:
            text += ' The camera slowly zoom in.'
        text = text_preprocessing(text, support_Chinese=self.support_Chinese)
        text = text if random.random() > self.cfg else ""
        return dict(pixel_values=image, caption=text)

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
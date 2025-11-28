import os
import math
import torch
import logging
import random
import subprocess
import numpy as np
import torch.distributed as dist

# from torch._six import inf
from torch import inf
from PIL import Image
from typing import Union, Iterable
import collections
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from opensora.model_variants.embeddings import get_1d_sincos_pos_embed_from_grid

from diffusers.utils import is_bs4_available, is_ftfy_available
import cv2
import html
import re
import urllib.parse as ul
import torch.nn.functional as F


if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

#######################################################################
#                             Pad and Resize                          #
#######################################################################
def pad_and_resize(img: torch.Tensor, pad_value: float = -1.0, resize_shape: int = None) -> torch.Tensor:
    if img.ndim == 4:
        B, C, H, W = img.shape
    elif img.ndim == 3:
        C, H, W = img.shape
        B = None
    else:
        raise ValueError("shape should be [B, C, H, W] or [C, H, W]")
    if H == W and resize_shape is None:
        return img

    diff = abs(H - W)
    if H < W:
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        padding = (0, 0, pad_top, pad_bottom)
    else:
        pad_left = diff // 2
        pad_right = diff - pad_left
        padding = (pad_left, pad_right, 0, 0)
    if B is not None:
        padded_img = F.pad(img, padding, mode='constant', value=pad_value)
        padded_img = padded_img.view(B, C, max(H, W), max(H, W))
    else:
        padded_img = F.pad(img.unsqueeze(0), padding, mode='constant', value=pad_value).squeeze(0)

    if resize_shape is not None:
        if B is not None:
            padded_img = F.interpolate(padded_img, (resize_shape, resize_shape), mode='bilinear', align_corners=False)
        else:
            padded_img = F.interpolate(padded_img.unsqueeze(0), (resize_shape, resize_shape), mode='bilinear', align_corners=False).squeeze(0)
    return padded_img

def center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    # 长图像crop上面部分
    if height > width:
        left = 0
        upper = 0
        right = width
        lower = width
    # 否则center crop
    else:
        left = (width - height) // 2
        upper = 0
        right = left + height
        lower = height
    return image.crop((left, upper, right, lower))

def get_face_masks_from_tensor(image_tensor, app, face_helper):
    image_np = (image_tensor.cpu().float().permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
    if image_np.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Input tensor must have 3 channels (C=3).")


    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # 使用 FaceAnalysis 进行人脸检测
    # image_info = app.get(image_bgr)
    bboxes, kpss = app.det_model.detect(image_bgr)


    # if len(image_info) > 0:
    #     # print("This is FaceAnalysis")
    #     # max det score
    #     info = image_info[0]
    #     x_1, y_1, x_2, y_2 = map(int, info['bbox'])
    #     # 计算正方形的边界
    #     side_length = max(x_2 - x_1, y_2 - y_1)
    #     center_x = (x_1 + x_2) // 2
    #     center_y = (y_1 + y_2) // 2
    #     x_1 = max(center_x - side_length // 2, 0)
    #     y_1 = max(center_y - side_length // 2, 0)
    #     x_2 = min(center_x + side_length // 2, width)
    #     y_2 = min(center_y + side_length // 2, height)
    #     face = image_np[y_1:y_2, x_1:x_2]
    if bboxes.shape[0] > 0:
        x_1, y_1, x_2, y_2 = map(int, bboxes[0, 0:4])
        # 计算正方形的边界
        side_length = max(x_2 - x_1, y_2 - y_1)
        center_x = (x_1 + x_2) // 2
        center_y = (y_1 + y_2) // 2
        x_1 = max(center_x - side_length // 2, 0)
        y_1 = max(center_y - side_length // 2, 0)
        x_2 = min(center_x + side_length // 2, width)
        y_2 = min(center_y + side_length // 2, height)
        face = image_np[y_1:y_2, x_1:x_2]

    else:
        face = np.zeros((384, 384, 3), dtype=np.uint8)
    return face


class GaussianNoiseAdder:
    def __init__(self, mean=-3.0, std=0.5, clear_ratio=0.05):
        self.mean = mean
        self.std = std
        self.clear_ratio = clear_ratio
    # pixel_values: (B, C, T, H, W)
    # mask: (B, 1, T, H, W)
    
    def add_noise(self, conditional_image, mask=None):
        if random.random() < self.clear_ratio:
            return conditional_image
        # black img
        if conditional_image.max() <= -0.99 or conditional_image.abs().max() < 1e-6:
            return conditional_image
        noise_sigma = torch.normal(mean=self.mean, std=self.std, size=(conditional_image.shape[0],), device=conditional_image.device)
        noise_sigma = torch.exp(noise_sigma).clamp(min=1e-6)  # 限制最小值，避免极端小值
        noise_sigma = torch.exp(noise_sigma).to(dtype=conditional_image.dtype)
        noise = torch.randn_like(conditional_image) * noise_sigma[:, None, None, None, None]
        #noise = torch.where(mask < 0.5, noise, torch.zeros_like(noise))
        if mask is not None:
            # mask: (B,1,T,H,W) → (B,C,T,H,W) 以便逐通道广播
            noise = noise * (mask < 0.5)

        return conditional_image + noise
        

def black_image(width, height):
    """generate a black image

    Args:
        width (int): image width
        height (int): image height

    Returns:
        _type_: a black image
    """
    black_image = Image.new("RGB", (width, height), (0, 0, 0))
    return black_image

def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

def find_model(model_name):
    """
    Finds a pre-trained Latte model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find Latte checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)

    # if "ema" in checkpoint:  # supports checkpoints from train.py
    #     print('Using Ema!')
    #     checkpoint = checkpoint["ema"]
    # else:
    print('Using model!')
    checkpoint = checkpoint['model']
    return checkpoint

#################################################################################
#                             Training Clip Gradients                           #
#################################################################################

def get_grad_norm(
        parameters: _tensor_or_tensors, norm_type: float = 2.0) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    return total_norm


def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, clip_grad=True) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)

    if clip_grad:
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        # gradient_cliped = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
        # print(gradient_cliped)
    return total_norm


def get_experiment_dir(root_dir, args):
    # if args.pretrained is not None and 'Latte-XL-2-256x256.pt' not in args.pretrained:
    #     root_dir += '-WOPRE'
    if args.use_compile:
        root_dir += '-Compile'  # speedup by torch compile
    if args.attention_mode:
        root_dir += f'-{args.attention_mode.upper()}'
    # if args.enable_xformers_memory_efficient_attention:
    #     root_dir += '-Xfor'
    if args.gradient_checkpointing:
        root_dir += '-Gc'
    if args.mixed_precision:
        root_dir += f'-{args.mixed_precision.upper()}'
    root_dir += f'-{args.max_image_size}'
    return root_dir

def get_precision(args):
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return dtype

#################################################################################
#                             Training Logger                                   #
#################################################################################

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            # format='[\033[34m%(asctime)s\033[0m] %(message)s',
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)

    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def create_tensorboard(tensorboard_dir):
    """
    Create a tensorboard that saves losses.
    """
    if dist.get_rank() == 0:  # real tensorboard
        # tensorboard
        writer = SummaryWriter(tensorboard_dir)

    return writer


def write_tensorboard(writer, *args):
    '''
    write the loss information to a tensorboard file.
    Only for pytorch DDP mode.
    '''
    if dist.get_rank() == 0:  # real tensorboard
        writer.add_scalar(args[0], args[1], args[2])


#################################################################################
#                      EMA Update/ DDP Training Utils                           #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            # os.environ["MASTER_PORT"] = "29566"
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


#################################################################################
#                             Testing  Utils                                    #
#################################################################################

def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros((t, (padding + h) * nrow + padding,
                              (padding + w) * ncol + padding, c), dtype=torch.uint8)

    print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]

    return video_grid


#################################################################################
#                             MMCV  Utils                                    #
#################################################################################


def collect_env():
    # Copyright (c) OpenMMLab. All rights reserved.
    from mmcv.utils import collect_env as collect_base_env
    from mmcv.utils import get_git_hash
    """Collect the information of the running environments."""

    env_info = collect_base_env()
    env_info['MMClassification'] = get_git_hash()[:7]

    for name, val in env_info.items():
        print(f'{name}: {val}')

    print(torch.cuda.get_arch_list())
    print(torch.version.cuda)


#################################################################################
#                          Pixart-alpha  Utils                                  #
#################################################################################


bad_punct_regex = re.compile(r'['+'#®•©™&@·º½¾¿¡§~'+'\)'+'\('+'\]'+'\['+'\}'+'\{'+'\|'+'\\'+'\/'+'\*' + r']{1,}')  # noqa

def text_preprocessing(text, support_Chinese=True):
    # The exact text cleaning as was in the training stage:
    text = clean_caption(text, support_Chinese=support_Chinese)
    text = clean_caption(text, support_Chinese=support_Chinese)
    return text

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def clean_caption(caption, support_Chinese=True):
    caption = str(caption)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub('<person>', 'person', caption)
    # urls:
    caption = re.sub(
        r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
        '', caption)  # regex for urls
    caption = re.sub(
        r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
        '', caption)  # regex for urls
    # html:
    caption = BeautifulSoup(caption, features='html.parser').text

    # @<nickname>
    caption = re.sub(r'@[\w\d]+\b', '', caption)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
    caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
    caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
    caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
    caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
    caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
    if not support_Chinese:
        caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)  # Chinese
    #######################################################

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
        '-', caption)

    # кавычки к одному стандарту
    caption = re.sub(r'[`´«»“”¨]', '"', caption)
    caption = re.sub(r'[‘’]', "'", caption)

    # &quot;
    caption = re.sub(r'&quot;?', '', caption)
    # &amp
    caption = re.sub(r'&amp', '', caption)

    # ip adresses:
    caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

    # article ids:
    caption = re.sub(r'\d:\d\d\s+$', '', caption)

    # \n
    caption = re.sub(r'\\n', ' ', caption)

    # "#123"
    caption = re.sub(r'#\d{1,3}\b', '', caption)
    # "#12345.."
    caption = re.sub(r'#\d{5,}\b', '', caption)
    # "123456.."
    caption = re.sub(r'\b\d{6,}\b', '', caption)
    # filenames:
    caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

    #
    caption = re.sub(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

    caption = re.sub(bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
    caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r'(?:\-|\_)')
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, ' ', caption)

    caption = basic_clean(caption)

    caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
    caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
    caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

    caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
    caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
    caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
    caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
    caption = re.sub(r'\bpage\s+\d+\b', '', caption)

    caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

    caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

    caption = re.sub(r'\b\s+\:\s+', r': ', caption)
    caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
    caption = re.sub(r'\s+', ' ', caption)

    caption.strip()

    caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
    caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
    caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
    caption = re.sub(r'^\.\S+$', '', caption)

    return caption.strip()

def get_condition(_x, cotracker, device, args):
    c, frames, height, width = _x.shape
    points = sample_random_points(0, height, width, 3500, args.sample_type)
    points = torch.tensor(points)[None].float().to(device)
    pred = []
    visiable = []
    with torch.no_grad():
        for _ in range(0, points.shape[1], 2000):
            _pred, _visiable = cotracker((_x[None].permute(0, 2, 1, 3, 4)+1)/2*255, grid_size=0, queries=points[:,_:_+2000,:])
            pred.append(_pred)
            visiable.append(_visiable)
    pred = torch.cat(pred, dim = 2)
    visiable = torch.cat(visiable, dim = 2)
    points_num = pred.shape[2]
    pos_embedding = get_1d_sincos_pos_embed_from_grid(args.condition_dim, torch.tensor(list(range(points_num))))
    indices = torch.randperm(points_num) # 打乱所有的点
    pos_embedding = pos_embedding[indices]
    condition = np.zeros((args.condition_dim, frames, height, width))
    for t in range(frames):
        for n in range(len(pos_embedding)):
            x_n = int(pred[0, t, n, 0]) # w
            y_n = int(pred[0, t, n, 1]) # h
            if x_n >= width or x_n < 0:
                continue
            if y_n >= height or y_n < 0:
                continue
            condition[:, t, y_n, x_n] = visiable[0, t, n].item() * pos_embedding[n,:]
    torch.cuda.empty_cache()
    return condition

def sample_patches(h, w, frame_idx, num_patches=49, patch_size=5):
    """
    从 h*w 的矩阵中均匀采样 num_patches 个块，每个块大小为 patch_size x patch_size，
    并返回这些块中所有点的坐标。

    :param h: 矩阵的高度
    :param w: 矩阵的宽度
    :param num_patches: 需要采样的小块数量，默认为 49
    :param patch_size: 每个小块的大小，默认为 5x5
    :return: 一个列表，包含 num_patches 个小块中所有点的坐标
    """
    # 计算每个网格的高度和宽度
    grid_height = h // int(np.sqrt(num_patches))
    grid_width = w // int(np.sqrt(num_patches))

    # 初始化结果列表
    patch_coords = []

    # 遍历每个网格
    for i in range(int(np.sqrt(num_patches))):
        for j in range(int(np.sqrt(num_patches))):
            # 计算当前网格的起始坐标
            grid_start_x = j * grid_width
            grid_start_y = i * grid_height
            
            # 确保网格内有足够的空间放置 patch
            if grid_start_x + patch_size <= w and grid_start_y + patch_size <= h:
                # 计算当前小块的起始坐标
                patch_start_x = grid_start_x + np.random.randint(0, grid_width - patch_size + 1)
                patch_start_y = grid_start_y + np.random.randint(0, grid_height - patch_size + 1)
                
                # 获取当前小块中所有点的坐标
                for x in range(patch_start_x, patch_start_x + patch_size):
                    for y in range(patch_start_y, patch_start_y + patch_size):
                        patch_coords.append((frame_idx, x, y))

    return patch_coords

def sample_random_points(frame_max_idx, height, width, num_points, sample_type):
    """Sample random points with (time, height, width) order."""
    if sample_type == 'random':
        y = np.random.randint(0, height, (num_points, 1))
        x = np.random.randint(0, width, (num_points, 1))
        t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
        points = np.concatenate((t, x, y), axis=-1).astype(
            np.int32
        )  # [num_points, 3]
        return points
    elif sample_type == 'grid':
        # 49 points + random shift
        points = sample_patches(height, width, 0, 49, 8)
        return points
    elif sample_type == 'sparse':
        points = sample_patches(height, width, 0, 16, 8)
        return points
        
if __name__ == '__main__':
    
    # caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
    a = "امرأة مسنة بشعر أبيض ووجه مليء بالتجاعيد تجلس داخل سيارة قديمة الطراز، تنظر من خلال النافذة الجانبية بتعبير تأملي أو حزين قليلاً."
    print(a)
    print(text_preprocessing(a))


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Optional
import json
import gc
import numpy as np
import random
from einops import rearrange
from tqdm import tqdm
from accelerate.state import AcceleratorState
try:
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import convert_unet_state_dict_to_peft
except:
    pass
# add for debug
import sys
import matplotlib.pyplot as plt


from opensora.adaptor.modules import replace_with_fp32_forwards
torch_npu = None
npu_config = None
from opensora.utils.parallel_states import initialize_sequence_parallel_state, \
    destroy_sequence_parallel_group, get_sequence_parallel_state, set_sequence_parallel_state
from opensora.utils.communications import prepare_parallel_data, broadcast

import time
from dataclasses import field, dataclass
from torch.utils.data import DataLoader
from copy import deepcopy
import accelerate
import torch
from torch.nn import functional as F
import torch.optim as optim
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMScheduler, PNDMScheduler, DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from opensora.dataset import getdataset
from opensora.models import CausalVAEModelWrapper
from opensora.models.text_encoder import get_text_enc, get_text_warpper
from opensora.models.causalvideovae import ae_stride_config, ae_channel_config
from opensora.models.causalvideovae import ae_norm, ae_denorm
from opensora.models.diffusion import Diffusion_models, Diffusion_models_class
from opensora.utils.dataset_utils import Collate2, LengthGroupedSampler
from opensora.utils.MultiResolutionSampler import VariableVideoBatchSampler
from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.train.diffusion import EDM, DDPM, PyramidFlowMatching, AutoRegressivePyramidFlowMatching, CogVideoXDDPM, CustomDiffusion, FlowMatching
from opensora.pyramid_dit.modeling_text_encoder import SD3TextEncoderWithMask
from opensora.pyramid_dit.modeling_pyramid_mmdit import PyramidDiffusionMMDiT

from opensora.flux_modules import (
    PyramidFluxTransformer,
    FluxTextEncoderWithMask,
)

from opensora.video_vae.modeling_causal_vae import CausalVideoVAE
from opensora.train.validation import get_validation_runner
from opensora.utils.utils import GaussianNoiseAdder



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
logger = get_logger(__name__)


def show_ref(img, output):
    input_to_display = img.cpu().float().numpy()
    if input_to_display.shape[0] == 1:  # 单通道图像
        input_to_display = input_to_display.squeeze(0)  # 转换为 (H, W)
    else:
        input_to_display = input_to_display.transpose(1, 2, 0)
    
    # [0, 1]
    if input_to_display.ndim != 2:
        input_to_display = (input_to_display + 1) / 2
    plt.imshow(input_to_display, cmap='gray' if input_to_display.ndim == 2 else None, vmin=0, vmax=1)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(output, bbox_inches='tight', pad_inches=0)  # 保存图像
    plt.close()

def show_train(x,condition,face_mask,ocr_mask,output):
    video_len = x.shape[1]
    t = np.random.randint(0, video_len)
    vae_t = (t + 3) // 4
    x_t = x[:, t, :, :].cpu().float().numpy()
    condition_t = condition[:, t, :, :].cpu().float().numpy()
    face_mask_t = face_mask[:, vae_t, :, :].squeeze(0).cpu().float().numpy()
    ocr_mask_t = ocr_mask[:, vae_t, :, :].squeeze(0).cpu().float().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    images = [x_t, condition_t, face_mask_t, ocr_mask_t]
    titles = [f'x_{t}', f'condition_{t}', f'face_mask_{vae_t}', f'ocr_mask_{vae_t}']
    cmaps = [None, None, 'gray', 'gray']
    for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        ax = axes[i]
        if img.ndim == 3: 
            img = img.transpose(1, 2, 0)
            img = (img + 1) / 2
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight', pad_inches=0)
    plt.close()

def serialize_config_recursively(config):
    allowed_types = (int, float, str, bool, torch.Tensor)
    if isinstance(config, dict):
        return {k: serialize_config_recursively(v) for k, v in config.items()}
    elif isinstance(config, (list, tuple)):
        return str(config)
    elif isinstance(config, allowed_types):
        return config
    else:
        return str(config)

def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


class ProgressInfo:
    def __init__(self, global_step, local_step=0, train_loss=0.0):
        self.global_step = global_step
        self.local_step = local_step
        self.train_loss = train_loss
        # log epoch loss
        self.train_loss_epoch = 0.0

        # track training speed
        self.move_data_cost = 0
        self.vae_encoder_cost = 0
        self.text_encoder_cost = 0
        self.forward_cost = 0
        self.backward_cost = 0
        self.step_total_cost = 0


#################################################################################
#                                  Training Loop                                #
#################################################################################

def custom_models(args, weight_dtype):
    from opensora.model_variants import get_dit
    from opensora.model_variants import get_controlnet
    from opensora.vae_variants import get_vae
    from opensora.encoder_variants import get_text_enc

    transformer, model_cls = get_dit(args.dit_name, args.config_path, weight_dtype)
    # controlnet, controlnet_model_cls = get_controlnet(args.controlnet_name, args.controlnet_config_path, weight_dtype)
    transformer.set_up_controlnet(args.controlnet_config_path, weight_dtype)
        # add ipadapter after load pretrained
    if args.ipadapter_config_path:
        transformer.set_up_ipadapter(args.ipadapter_config_path, weight_dtype)
    if args.refextractor_config_path:
        transformer.set_up_refextractor(args.refextractor_config_path, weight_dtype)
    
    
    transformer.gradient_checkpointing = args.gradient_checkpointing

    if args.gradient_checkpointing:
        try:
            transformer.enable_gradient_checkpointing()
        except Exception as e:
            print(e)
        assert transformer.gradient_checkpointing
        
    # import ipdb; ipdb.set_trace()
    if args.load_encoders:
        text_encoder = get_text_enc(args.text_encoder_name, args.text_encoder_path, weight_dtype)
        text_encoder.eval()
        text_encoder.requires_grad_(False)

        vae = get_vae(args.vae_name, args.vae_path, weight_dtype)
        vae.eval()
        # vae.enable_tiling()
        try:
            vae.enable_tiling()
            vae.enable_slicing()
        except Exception as e:
            pass
    else:
        text_encoder = None
        vae = None
    
    return text_encoder, vae, transformer, model_cls
    

def get_parameter_groups(model, weight_decay=1e-5, base_lr=1e-4, skip_list=(), get_num_layer=None, get_layer_scale=None, **kwargs):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue

        default_scale=1.
        
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list: # param.ndim <= 1 len(param.shape) == 1
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = default_scale

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": base_lr,
                "lr_scale": scale,
            }

            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": base_lr,
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, **kwargs):
    opt_lower = args.optimizer.lower()
    weight_decay = args.adam_weight_decay

    skip = {}
    if skip_list is not None:
        skip = skip_list
    elif hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    print(f"Skip weight decay name marked in model: {skip}")
    parameters = get_parameter_groups(model, weight_decay, args.learning_rate, skip, get_num_layer, get_layer_scale, **kwargs)
    weight_decay = 0.

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.learning_rate, weight_decay=weight_decay)
    if hasattr(args, 'adam_epsilon') and args.adam_epsilon is not None:
        opt_args['eps'] = args.adam_epsilon
    if hasattr(args, 'adam_beta1') and args.adam_beta1 is not None:
        opt_args['betas'] = (args.adam_beta1, args.adam_beta2)
    
    print('Optimizer config:', opt_args)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    # use LayerNorm, GeLu, SiLu always as fp32 mode
    if args.enable_stable_fp32:
        replace_with_fp32_forwards()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # 单卡再用
    # from accelerate import DistributedDataParallelKwargs
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs],
    )
    accelerator.even_batches=False
    try:
        AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']=1
    except:
        pass

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # Setup data:
    setattr(args, "valid", False)
    args.valid=False
    train_dataset = getdataset(args)
    sampler = VariableVideoBatchSampler(
        train_dataset,
        world_size=accelerator.num_processes,
        drop_last=True,
        verbose=True,
        shuffle=True,
        num_bucket_build_workers=10,
        train_fps = args.train_fps,
        valid = args.valid,
        seed = int(args.seed),
        bucket_name = args.bucket_name
    )
    train_dataloader = DataLoader(
        train_dataset,
        # pin_memory=True,
        collate_fn=Collate2(args), #collate：有些分辨率可能不是16的整数倍，padding一下
        num_workers=args.dataloader_num_workers,
        batch_sampler=sampler,
        # prefetch_factor=4
    )

    valid_args = deepcopy(args)
    valid_args.data = args.valid_data
    valid_args.valid = True
    valid_dataset = getdataset(valid_args)
    valid_sampler = VariableVideoBatchSampler(
        valid_dataset,
        world_size=accelerator.num_processes,
        drop_last=True,
        verbose=True,
        shuffle=False,
        num_bucket_build_workers=10,
        train_fps = args.train_fps,
        valid = valid_args.valid,
        bucket_name = args.bucket_name
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        collate_fn=Collate2(args), #collate：有些分辨率可能不是16的整数倍，padding一下
        num_workers=args.dataloader_num_workers,
        batch_sampler=valid_sampler,
    )
    logger.info(f"valid size: {len(valid_dataloader)}")
    validation_runner = get_validation_runner(args)
    logger.info(f'after train_dataloader')

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    text_enc, vae, model, dit_model_cls = custom_models(args, weight_dtype)
    

    # Check if controlnet weights were loaded from pretrained
    # not load controlnet in init_from_transformer
    controlnet_loaded = False
    refextractor_loaded= False

    # use pretrained dit?
    if args.pretrained:
        print(f"loading from {args.pretrained}")
        model_state_dict = model.state_dict()
        # import ipdb;ipdb.set_trace()
        if args.pretrained.endswith(".safetensors"):  # pixart series
            from safetensors.torch import load_file as safe_load
            pretrained_checkpoint = safe_load(args.pretrained, device="cpu")
            pretrained_keys = set(list(pretrained_checkpoint.keys()))
            model_keys = set(list(model_state_dict.keys()))
            common_keys = list(pretrained_keys & model_keys)
            checkpoint = {k: pretrained_checkpoint[k] for k in common_keys if model_state_dict[k].numel() == pretrained_checkpoint[k].numel()}
        elif os.path.isdir(args.pretrained):
            checkpoint = {}
            shard_files_safe = [f for f in os.listdir(args.pretrained) if f.endswith('.safetensors')]
            shard_files_bin = [f for f in os.listdir(args.pretrained) if f.endswith('.bin')]
            # safe load
            if len(shard_files_safe) > 0:
                from safetensors.torch import load_file as safe_load
                for shard_file in shard_files_safe:
                    state_dict = safe_load(os.path.join(args.pretrained, shard_file), device='cpu')
                    checkpoint.update(state_dict)
            
            # torch_load
            elif len(shard_files_bin) > 0:
                for shard_file in shard_files_bin:
                    state_dict = torch.load(os.path.join(args.pretrained, shard_file), map_location='cpu')
                    checkpoint.update(state_dict)
        else:  # latest stage training weight
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            if 'module' in checkpoint:
                checkpoint = checkpoint['module']
        try:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print("!!!!!!!! error in load pretrained ckpt:", e)
        if accelerator.is_main_process:
            accelerator.print(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
            accelerator.print(f'Successfully load {len(model_state_dict) - len(missing_keys)}/{len(model_state_dict)} keys from {args.pretrained}!')

        if any('controlnet' in key for key in checkpoint.keys()):
            controlnet_loaded = True
        if any('refextractor' in key for key in checkpoint.keys()):
            refextractor_loaded = True

    if args.pretrained_controlnet:
        print(f"loading from {args.pretrained_controlnet}")
        model_state_dict = model.controlnet.state_dict()
        if args.pretrained_controlnet.endswith(".safetensors"):  # pixart series
            from safetensors.torch import load_file as safe_load
            # import ipdb;ipdb.set_trace()
            pretrained_checkpoint = safe_load(args.pretrained, device="cpu")
            pretrained_keys = set(list(pretrained_checkpoint.keys()))
            model_keys = set(list(model_state_dict.keys()))
            common_keys = list(pretrained_keys & model_keys)
            checkpoint = {k: pretrained_checkpoint[k] for k in common_keys if model_state_dict[k].numel() == pretrained_checkpoint[k].numel()}
        elif os.path.isdir(args.pretrained_controlnet):
            checkpoint = {}
            shard_files_safe = [f for f in os.listdir(args.pretrained_controlnet) if f.endswith('.safetensors')]
            shard_files_bin = [f for f in os.listdir(args.pretrained_controlnet) if f.endswith('.bin')]
            # safe load
            if len(shard_files_safe) > 0:
                from safetensors.torch import load_file as safe_load
                for shard_file in shard_files_safe:
                    state_dict = safe_load(os.path.join(args.pretrained_controlnet, shard_file), device='cpu')
            
            # torch_load
            elif len(shard_files_bin) > 0:
                for shard_file in shard_files_bin:
                    state_dict = torch.load(os.path.join(args.pretrained_controlnet, shard_file), map_location='cpu')
                    checkpoint.update(state_dict)
        else:  # latest stage training weight
            checkpoint = torch.load(args.pretrained_controlnet, map_location='cpu')
        try:
            missing_keys, unexpected_keys = model.controlnet.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print("!!!!!!!! error in load pretrained ckpt:", e)
        if accelerator.is_main_process:
            accelerator.print(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
            accelerator.print(f'Successfully load {len(model_state_dict) - len(missing_keys)}/{len(model_state_dict)} keys from {args.pretrained_controlnet}!')
    
    if args.init_from_transformer:
        if controlnet_loaded:
            if accelerator.is_main_process:
                accelerator.print(f'controlnet_loaded!!, dont init_from_transformer!!')

        else:
            controlnet_state_dict = {}
            for name, params in model.state_dict().items():
                if 'refextractor' in name or 'controlnet' in name:
                    continue
                controlnet_state_dict[name] = params
            m, u = model.controlnet.load_state_dict(controlnet_state_dict, strict=False)
            if accelerator.is_main_process:
                accelerator.print(f'[ Weights from transformer was loaded into controlnet ] [M: {m} | U: {u}]')

        if refextractor_loaded:
            if accelerator.is_main_process:
                accelerator.print(f'refextractor_loaded!!, dont init_from_transformer!!')
        else:
            refextractor_state_dict = {}
            special_key = "patch_embedding.weight"
            for name, params in model.state_dict().items():
                if 'refextractor' in name or 'controlnet' in name:
                    continue
                if name == special_key:
                    pretrained_weight = params
                    current_weight = model.refextractor.state_dict()[special_key]
                    if pretrained_weight.shape[1] == 16 and current_weight.shape[1] == 32:
                        if accelerator.is_main_process:
                            accelerator.print(f"Special handling for '{special_key}' with torch.cat: extending {pretrained_weight.shape[1]} to {current_weight.shape[1]}.")
                        new_channel_shape = list(pretrained_weight.shape)
                        new_channel_shape[1] = 16
                        zero_channels = pretrained_weight.new_zeros(new_channel_shape)
                        adjusted_weight = torch.cat([pretrained_weight, zero_channels], dim=1)
                        refextractor_state_dict[special_key] = adjusted_weight
                        continue
                refextractor_state_dict[name] = params
        
            m, u = model.refextractor.load_state_dict(refextractor_state_dict, strict=False)
            if accelerator.is_main_process:
                accelerator.print(f'[ Weights from transformer was loaded into refextractor ] [M: {m} | U: {u}]')



    # Freeze vae and text encoders.
    if args.load_encoders:
        vae.requires_grad_(False)
        text_enc.requires_grad_(False)
        vae.to(accelerator.device, dtype=torch.float32)
        text_enc.to(accelerator.device, dtype=weight_dtype)
    model.to(accelerator.device, dtype=weight_dtype)
    # controlnet.to(accelerator.device, dtype=weight_dtype)
    # Set model as trainable.
    model.requires_grad_(False)
    # set ip_adapter trainable
    model.train()
    
    #可以这么实现，但是肯定不是最好的实现方法
    if args.only_patch_embeding:
        for k, v in model.named_parameters():
            if 'img_in' in k:
                v.requires_grad = True
    if args.training_modules:
        if "all" in args.training_modules:
            model.requires_grad_(True)
        else:
            for name, param in model.named_parameters():
                for module in args.training_modules:
                    if module in name:
                        param.requires_grad = True
    # 最后再设置，避免混淆
    # if args.frozen_controlnet:
    #     model.controlnet.requires_grad_(False)
    # else:
    #     model.controlnet.requires_grad_(True)
    
    # model.controlnet.enable_refuser_params_grad()
    # model.enable_refuser_params_grad()
    model.refextractor.requires_grad_(True)
    model.controlnet.requires_grad_(True)


    if args.diffusion_formula == 'DDPM':
        diffusion = DDPM(args.reflow)
    elif args.diffusion_formula == 'EDM':
        diffusion = EDM()
    elif args.diffusion_formula == 'CogVideo':
        diffusion = CustomDiffusion()
    elif args.diffusion_formula == 'AutoregressivePyramidFlow':
        stages = args.stages
        stage_range = np.linspace(0,1,stages+1)
        diffusion = AutoRegressivePyramidFlowMatching(
            stages = stages,
            stage_range = stage_range,
            frame_per_unit = args.frame_per_unit,
            max_history_len = args.max_history_len,
            extra_sample_steps = args.extra_sample_steps,
            model_name = args.model_type,
            process_index = accelerator.process_index,
            max_unit_num = int(args.max_unit_num)
        )
    elif args.diffusion_formula == 'PyramidFlowMatching':
        diffusion = PyramidFlowMatching()
    elif args.diffusion_formula == 'FlowMatching':
        diffusion = FlowMatching()
    # Create EMA for the unet.
    if args.use_ema:
        assert not args.lora
        ema_model = deepcopy(model)
        ema_model = EMAModel(ema_model.parameters(), decay=args.ema_decay, update_after_step=args.ema_start_step,
                             model_cls=dit_model_cls, model_config=ema_model.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "model_ema"))
                # if models:
                #     print("len(models)",len(models))
                #     final_model = models[-1]
                #     final_model = unwrap_model(accelerator,final_model)
                #     final_model.save_pretrained(os.path.join(output_dir, "fp32_model"))
                # for i, model in enumerate(models): #两个模型，和prepare中的顺序一致
                #     model.save_pretrained(os.path.join(output_dir, "model"))
                #     if weights:  # Don't pop if empty
                #         # make sure to pop weight so that corresponding model is not saved again
                #         weights.pop()
                # models[0].save_pretrained(os.path.join(output_dir, "model"))
                # if weights:  # Don't pop if empty
                #     weights.pop()
                # weights.pop()

        def load_model_hook(models, input_dir): #当使用deepspeed进行训练的时候，这里models是空的
            #TODO: 模型尺寸太大的时候可能出现问题
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "model_ema"), dit_model_cls)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            # for i in range(len(models)):
            #     # pop models so that they are not loaded again
            #     model = models.pop()
            #     # load diffusers style into model
            #     load_model = dit_model_cls.from_pretrained(input_dir, subfolder="model")
            #     model.register_to_config(**load_model.config)
            #     model.load_state_dict(load_model.state_dict())
            #     del load_model
            # if len(models) > 0:
            #     # models.pop()
            #     controlnet = models.pop()
            #     load_model = controlnet_model_cls.from_pretrained(input_dir, subfolder='controlnet')
            #     controlnet.register_to_config(**load_model.config)
            #     controlnet.load_state_dict(load_model.state_dict())
            #     del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * accelerator.num_processes
        )
    params_to_optimize = model.parameters()
    # params_to_optimize = [param for param in model.parameters() if param.requires_grad]
    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")
        optimizer_class = prodigyopt.Prodigy
        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
    # import pdb;pdb.set_trace()
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, #NOTE: 原始代码在这里计入了accumulate，但是不应该计入的
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    if accelerator.is_main_process:
        accelerator.print(f'before accelerator.prepare')
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
        accelerator.print(f"  Total training parameters = {total_params} B")
        # 将训练参数写入 args.output_dir
        with open(f"{args.output_dir}/training_parameters.txt", "w") as f:
            f.write(f"Total training parameters = {total_params} B\n")
            f.write("Trained parameter names:\n")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    f.write(f"{name}\n")

    model, optimizer, train_dataloader, lr_scheduler, valid_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, valid_dataloader
    )
    if accelerator.is_main_process:
        accelerator.print(f'after accelerator.prepare')
        accelerator.print(f"  Total training parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B")

    if args.use_ema:
        ema_model.to(accelerator.device)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        config = serialize_config_recursively(vars(args))
        accelerator.init_trackers(os.path.basename(args.output_dir), config=config)

    # Train!
    total_batch_size = accelerator.num_processes * args.gradient_accumulation_steps
    total_batch_size = total_batch_size

    if accelerator.is_main_process:
        accelerator.print("***** Running training *****")
        accelerator.print(f"  Model = {model}")
        accelerator.print(f"  Num examples = {len(train_dataset)}")
        accelerator.print(f"  Num Epochs = {args.num_train_epochs}")
        accelerator.print(f"  Num update steps per epoch = {num_update_steps_per_epoch}")
        accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        accelerator.print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
        # accelerator.print(f"  ControlNet Trainable Parameters: {controlnet_and_ipadapter_counts['controlnet_params'] / 1e9:.3f} B")
        # accelerator.print(f"  IP-Adapter Trainable Parameters: {controlnet_and_ipadapter_counts['ipadapter_params'] / 1e9:.3f} B")
        accelerator.print(f"  Total training parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.3f} B")

    global_step = 0
    local_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            try:
                local_step = int(path.split("-")[2])
            except:
                local_step = 0

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    if not args.start_from_middle:
        local_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_info = ProgressInfo(global_step, local_step, train_loss=0.0)

    def sync_gradients_info(loss, input_shape,conditioning_scale):
        # Checks if the accelerator has performed an optimization step behind the scenes
        # NOTE: 这里log的是当前进程上的loss，而非gather之后的平均loss
        if torch.isnan(loss):
            raise RuntimeError("loss is nan!!!!!!!, terminating")
        if args.use_ema:
            ema_model.step(model.parameters())
        progress_bar.update(1)
        progress_info.global_step += 1
        progress_info.local_step += 1
        end_time = time.time()
        one_step_duration = end_time - start_time
        # 放下面一起log
        # accelerator.log({"train_loss": progress_info.train_loss}, step=progress_info.global_step) # gather之后的平均loss在这里log
        # progress_info.train_loss = 0.0

        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
        if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
            if progress_info.global_step % args.checkpointing_steps == 0 or progress_info.local_step == num_update_steps_per_epoch - 1:
                try:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    checkpoint_path_override = None 
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        accelerator.wait_for_everyone()
                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            if accelerator.is_main_process:
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints[:-1]:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                            checkpoint_path_override = os.path.join(args.output_dir, removing_checkpoints[-1])
                        else:    
                            checkpoint_path_override = None
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{progress_info.global_step}-{progress_info.local_step}")
                    if checkpoint_path_override is not None:
                        if accelerator.is_main_process and checkpoint_path_override != save_path:
                            os.rename(checkpoint_path_override, save_path)
                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                except Exception as e:
                    print("attempted to save ckpts, caught error:", e)
                    print('wait until next turn')

        progress_info.train_loss_epoch += progress_info.train_loss
        logs = {
            "train_loss": progress_info.train_loss,
            "step_loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "conditioning_scale":conditioning_scale,
            "input_shape": str(input_shape),
        }
        accelerator.log(logs, step=progress_info.global_step)
        progress_info.train_loss = 0.0

        progress_bar.set_postfix(**logs)

    def run(model_input, condition, model_kwargs, prof):
        global start_time
        start_time = time.time()
        
        stage_index = progress_info.global_step % args.stages

        loss = diffusion.get_loss_wanx(condition, model, model_input, model_kwargs, args, accelerator = accelerator, stage_index = stage_index)

        #TODO: filter极端的loss
        # if loss < 0.01 or loss > 0.1:
        #     print(f'!!!!!!!!! loss is strange, do not update, value {loss}')
        #     loss *= 0 #TODO: 应该不是一个好方法，因为会影响到Adam的内部状态更新

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(1)).mean()
        progress_info.train_loss += avg_loss.detach().item() / args.gradient_accumulation_steps

        # Backpropagate
        accelerator.backward(loss)
        if accelerator.sync_gradients: #预期上，当设置了accum的时候，只有当进行参数更新的时候才会进入
            params_to_clip = model.parameters()
            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        optimizer.step() #acclerator会自动处理梯度累计
        lr_scheduler.step()
        optimizer.zero_grad()
        if accelerator.sync_gradients:# 预期上，当设置了accum的时候，只有当进行参数更新的时候才会进入
            if isinstance(model_input, tuple):
                model_input = model_input[0]
            sync_gradients_info(loss, model_input.shape,model_kwargs['conditioning_scale'])
            if args.do_valid and (progress_info.global_step == 1 or progress_info.global_step % args.evaluation_every == 0) and hasattr(diffusion, 'get_valid_loss'):
                #多卡进行validation
                if accelerator.is_local_main_process:
                    logger.info("==================> doing validation")
                validation_runner.run_valid(model, vae, text_enc, diffusion, valid_dataloader, accelerator, weight_dtype, args, logger)
                
        if prof is not None:
            prof.step()

        return loss

    def train_one_step(step_, data_item_, prof_=None):
        t1 = time.time()
        train_loss = 0.0
        reference_frame_mask = None
        x, caption, batch_embed_scale, condition, reference_frame, face_mask, ocr_mask, reference_pose = data_item_
        if reference_frame.shape[1] == 4: #BCTHW
            reference_frame, reference_frame_mask = torch.split(reference_frame, [3,1], dim=1)


        assert not torch.any(torch.isnan(x)), 'torch.any(torch.isnan(x))'
        x = x.to(accelerator.device, dtype=weight_dtype)  # B C T+num_images H W, 16 + 4
        face_mask.to(accelerator.device)
        ocr_mask.to(accelerator.device)
        # given video sqn and restore,
        # 使用restore controlnet的时候，采用vae,
        if args.restore_video:
            if "restore" in args.controlnet_config_path:
                condition = vae.encode(x.to(dtype = vae.dtype))
            else:
                # 否则采用mini encoder
                condition = x
        def is_ref_black(reference_frame):
            return reference_frame.max() <= -0.99 or reference_frame.abs().max() < 1e-6 

        conditioning_scale = 0 if is_ref_black(condition) else 1
        condition = condition.to(accelerator.device, dtype=weight_dtype)
        reference_pose = reference_pose.to(accelerator.device, dtype=weight_dtype)
        
        # input_image = x.clone()
        image_emb = None
        noise_adder = GaussianNoiseAdder(mean=-3.0, std=0.5, clear_ratio=args.conditional_clear_ratio)
        # 50% add_noise
        reference_frame = noise_adder.add_noise(reference_frame,reference_frame_mask)

        # use ori_reference_frame to produce face
        image = reference_frame.squeeze(2) #BCHW

        if reference_frame_mask is not None:
            reference_frame_mask = reference_frame_mask.repeat(1, 3, 1, 1, 1)
            reference_frame_mask = reference_frame_mask * 2.0 - 1.0

        # check input 
        if accelerator.is_main_process:
            if progress_info.global_step % 50 == 0:
                show_ref(image[0],os.path.join(args.output_dir,'sample_reference.png')) #CHW
                if reference_frame_mask is not None:
                    show_ref(reference_frame_mask.squeeze(2)[0],os.path.join(args.output_dir,'sample_reference_mask.png'))
                    show_ref(reference_pose.squeeze(2)[0],os.path.join(args.output_dir,'sample_reference_pose.png'))
                show_train(x[0],condition[0],face_mask[0],ocr_mask[0],os.path.join(args.output_dir,'sample_train.png'))

        # if args.diffusion_formula == 'CogVideo': #固定帧数和分辨率
        #     assert input_image.shape[-2] == args.max_height and input_image.shape[-1] == args.max_width

        t2 = time.time()
        # print('moving data cost:', t2 - t1)
        
        with torch.no_grad():
            if not args.load_cached_latents:
                assert args.load_encoders
                caption = [caption] if isinstance(caption, str) else caption
                cond = text_enc(caption, device = accelerator.device, max_length=args.max_sequence_length)
                x = x.to(accelerator.device, dtype=vae.dtype) 
                reference_frame = reference_frame.to(accelerator.device, dtype = vae.dtype)
                image_latents = vae.encode(reference_frame).to(weight_dtype)

                if args.task == 'i2v':
                    x = vae.encode(x) #already been scaled
                else:
                    x = vae.encode(x) #already been scaled
            else:
                cond, cond_mask, pooled_projections = caption
                x = x
                if args.task == 'i2v': #TODO: currently there is no noise and dropout
                    image_latents = x[:, :, :1].clone()

        t3 = time.time()
        # print('encoders cost:', t3 - t2)
        # print("input_shape:", input_image.shape, "latent shape:", x.shape)
        current_step_frame = x.shape[2]

        with accelerator.accumulate(model):
            assert not torch.any(torch.isnan(x)), 'after vae'
            if args.task == 't2v':
                if reference_frame_mask is not None:
                    reference_frame_mask = vae.encode(reference_frame_mask).to(weight_dtype)
                    image_latents = torch.concat([image_latents,reference_frame_mask],dim=1)
                x = x.to(weight_dtype)
                model_kwargs = dict(encoder_hidden_states=cond, batch_embed_scale=batch_embed_scale,
                                    image_emb=image_emb,image_latents=image_latents,face_mask=face_mask,ocr_mask=ocr_mask,conditioning_scale=conditioning_scale,token_replace_prob=args.token_replace_prob,image_pose=reference_pose)
                run(x, condition, model_kwargs, prof_)
            elif args.task == 'i2v':
                x = x.to(weight_dtype)
                x = (x, image_latents)
                model_kwargs = dict(encoder_hidden_states=cond, batch_embed_scale=batch_embed_scale,
                                    image_emb=image_emb,image_latents=image_latents,face_mask=face_mask,ocr_mask=ocr_mask,conditioning_scale=0)
                run(x, condition, model_kwargs, prof_)

        if progress_info.global_step >= args.max_train_steps:
            return True
        t4 = time.time()
        # print('model cost:', t4 - t3)
        return False

    def train_all_epoch(prof_=None):

        for epoch in range(first_epoch, args.num_train_epochs):
            progress_info.train_loss_epoch = 0.0  # 初始化每个 epoch 的损失
            num_steps_in_epoch = 0 # local step可能不是从0开始，用一个新的变量记录

            progress_info.train_loss = 0.0
            if progress_info.global_step >= args.max_train_steps:
                return True
            if progress_info.local_step > 0 or args.skip_extra > 0:
                print(f'skip first {progress_info.local_step * args.gradient_accumulation_steps} batches')
                print(f'skip {args.skip_extra} extra batches for sake of stablizing traing' )
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, progress_info.local_step * args.gradient_accumulation_steps + args.skip_extra)
                for step, data_item in enumerate(skipped_dataloader):
                    num_steps_in_epoch += 1
                    if train_one_step(step, data_item, prof_):
                        break
            else:
                for step, data_item in enumerate(train_dataloader):
                    num_steps_in_epoch += 1
                    if train_one_step(step, data_item, prof_):
                        break

            logs = {
                "epoch_loss": progress_info.train_loss_epoch / num_steps_in_epoch,
            }
            accelerator.log(logs, step=epoch)
            progress_info.local_step = 0 #每个epoch结束的时候，local step置为0

    train_all_epoch()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    from opensora.train.options import get_parser
    parser = get_parser()
    args = parser.parse_args()
    if args.lora:
        args.use_ema = False
    main(args)

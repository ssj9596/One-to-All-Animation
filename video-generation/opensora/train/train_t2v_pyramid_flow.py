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
import gc
import numpy as np
from einops import rearrange
from tqdm import tqdm
from accelerate.state import AcceleratorState

from opensora.adaptor.modules import replace_with_fp32_forwards
from opensora.pyramid_dit.modeling_pyramid_mmdit import PyramidDiffusionMMDiT
from opensora.pyramid_dit.modeling_text_encoder import SD3TextEncoderWithMask
from opensora.video_vae.modeling_causal_vae import CausalVideoVAE


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

from opensora.dataset import getdataset
from opensora.models import CausalVAEModelWrapper
from opensora.models.text_encoder import get_text_enc, get_text_warpper
from opensora.models.causalvideovae import ae_stride_config, ae_channel_config
from opensora.models.causalvideovae import ae_norm, ae_denorm
from opensora.models.diffusion import Diffusion_models, Diffusion_models_class
from opensora.utils.dataset_utils import Collate, LengthGroupedSampler
from opensora.utils.MultiResolutionSampler import VariableVideoBatchSampler
from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.train.diffusion import EDM, DDPM, PyramidFlowMatching

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
logger = get_logger(__name__)


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

class ProgressInfo:
    def __init__(self, global_step, local_step=0, train_loss=0.0):
        self.global_step = global_step
        self.local_step = local_step
        self.train_loss = train_loss

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

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    # use LayerNorm, GeLu, SiLu always as fp32 mode
    if args.enable_stable_fp32:
        replace_with_fp32_forwards()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    accelerator.even_batches=False
    try:
        AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']=1
    except:
        pass
    if args.num_frames != 1 and args.use_image_num == 0:
        initialize_sequence_parallel_state(args.sp_size)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
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

    # Create model:
    
    vae = CausalVideoVAE.from_pretrained(args.ae_path, torch_dtype=weight_dtype, interpolate=False).eval()
    vae.enable_tiling()

    # TODO: refactor in the future
    args.ae_stride_t = 8
    args.ae_stride_h = 8
    args.ae_stride_w = 8
    args.ae_stride = 8
    args.patch_size = 2
    args.patch_size_t = 1
    args.patch_size_h = 2
    args.patch_size_w = 2
    args.stride_t = 4
    args.stride = 16

    # Setup data:
    train_dataset = getdataset(args)
    sampler = VariableVideoBatchSampler(
        train_dataset,
        world_size=accelerator.num_processes,
        drop_last=True,
        verbose=True,
        shuffle=True,
        num_bucket_build_workers=10
    )
    train_dataloader = DataLoader(
        train_dataset,
        # pin_memory=True,
        collate_fn=Collate(args), #collate：有些分辨率可能不是16的整数倍，padding一下
        num_workers=args.dataloader_num_workers,
        batch_sampler=sampler,
        # prefetch_factor=4
    )

    logger.info(f'after train_dataloader')

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps * args.sp_size / args.train_sp_batch_size)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    # TODO: need to be improved
    model = PyramidDiffusionMMDiT.from_pretrained(
                args.pretrained_model_name_or_path, torch_dtype=torch_dtype, 
                use_gradient_checkpointing=args.gradient_checkpointing, 
                use_flash_attn=False, use_t5_mask=True,
                add_temp_pos_embed=True, temp_pos_embed_type='rope', 
                use_temporal_causal=True, interp_condition_pos=True,
            )
    text_enc = SD3TextEncoderWithMask(args.text_encode_name, torch_dtype=weight_dtype).eval()

    # Freeze vae and text encoders.
    vae.requires_grad_(False)
    text_enc.requires_grad_(False)
    model.train()
    
    assert args.diffusion_formula == 'PyramidFlow'
    if args.diffusion_formula == 'DDPM':
        diffusion = DDPM(args.reflow)
    elif args.diffusion_formula == 'EDM':
        diffusion = EDM()
    elif args.diffusion_formula == 'PyramidFlow':
        stages = args.stages
        stage_range = np.linspace(0,1, stages+1)
        diffusion = PyramidFlowMatching(
            stages = args.stages,
            stage_range = stage_range
        )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)
    # ae.vae.to(accelerator.device, dtype=weight_dtype)
    text_enc.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_model = deepcopy(model)
        ema_model = EMAModel(ema_model.parameters(), decay=args.ema_decay, update_after_step=args.ema_start_step,
                             model_cls=PyramidDiffusionMMDiT, model_config=ema_model.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "model_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "model"))
                    if weights:  # Don't pop if empty
                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "model_ema"), PyramidDiffusionMMDiT)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = PyramidDiffusionMMDiT.from_pretrained(input_dir, subfolder="model")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    params_to_optimize = model.parameters()
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
    logger.info(f"optimizer: {optimizer}")
    

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps * accelerator.num_processes, #实际上这些计算都不是很精准，不应该计入gradient_accumlation的
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    # model.requires_grad_(False)
    # model.pos_embed.requires_grad_(True)
    logger.info(f'before accelerator.prepare')
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    logger.info(f'after accelerator.prepare')
    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps * args.sp_size / args.train_sp_batch_size)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(os.path.basename(args.output_dir), config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    total_batch_size = total_batch_size // args.sp_size * args.train_sp_batch_size #这个的计算是没问题的
    logger.info("***** Running training *****")
    logger.info(f"  Model = {model}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num update steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total training parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B")
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

    def sync_gradients_info(loss, input_shape):
        # Checks if the accelerator has performed an optimization step behind the scenes
        if torch.isnan(loss):
            raise RuntimeError("loss is nan!!!!!!!, terminating")
        if args.use_ema:
            ema_model.step(model.parameters())
        progress_bar.update(1)
        progress_info.global_step += 1
        progress_info.local_step += 1
        end_time = time.time()
        one_step_duration = end_time - start_time
        accelerator.log({"train_loss": progress_info.train_loss}, step=progress_info.global_step)
        progress_info.train_loss = 0.0

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

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "input_shape": input_shape}
        progress_bar.set_postfix(**logs)

    def run(model_input, model_kwargs, prof):
        global start_time
        start_time = time.time()
         
        stage_index = progress_info.global_step % args.stages
        loss = diffusion.get_loss(model, model_input, model_kwargs, args, stage_index = stage_index)

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
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
            sync_gradients_info(loss, model_input.shape)
        if accelerator.is_main_process:
            if progress_info.global_step % args.checkpointing_steps == 0:
                if args.enable_tracker:
                    log_validation(args, model, ae, text_enc.text_enc, train_dataset.tokenizer, accelerator,
                                   weight_dtype, progress_info.global_step)
        if prof is not None:
            prof.step()


        return loss

    def train_one_step(step_, data_item_, prof_=None):
        train_loss = 0.0
        x, attn_mask, input_ids, cond_mask = data_item_
        if args.group_frame or args.group_resolution:
            if not args.group_frame:
                each_latent_frame = torch.any(attn_mask.flatten(-2), dim=-1).int().sum(-1).tolist()
                # logger.info(f'rank: {accelerator.process_index}, step {step_}, special batch has attention_mask '
                #             f'each_latent_frame: {each_latent_frame}')
                print(f'rank: {accelerator.process_index}, step {step_}, special batch has attention_mask '
                            f'each_latent_frame: {each_latent_frame}')
        assert not torch.any(torch.isnan(x)), 'torch.any(torch.isnan(x))'
        x = x.to(accelerator.device, dtype=ae.vae.dtype)  # B C T+num_images H W, 16 + 4

        attn_mask = attn_mask.to(accelerator.device)  # B T+num_images H W
        input_ids = input_ids.to(accelerator.device)  # B 1+num_images L
        cond_mask = cond_mask.to(accelerator.device)  # B 1+num_images L
        # if accelerator.process_index == 0:
        #     logger.info(f'rank: {accelerator.process_index}, x: {x.shape}, attn_mask: {attn_mask.shape}')
        input_image = x.clone()
        with torch.no_grad():
            B, N, L = input_ids.shape  # B 1+num_images L
            # use batch inference
            input_ids_ = input_ids.reshape(-1, L)
            cond_mask_ = cond_mask.reshape(-1, L)
            cond = text_enc(input_ids_, cond_mask_)  # B 1+num_images L D
            cond = cond.reshape(B, N, L, -1)
            # Map input images to latent space + normalize latents
            x = ae.encode(x)  # B C T H W

            # def custom_to_video(x: torch.Tensor, fps: float = 2.0, output_file: str = 'output_video.mp4') -> None:
            #     from examples.rec_video import array_to_video
            #     x = x.detach().cpu()
            #     x = torch.clamp(x, -1, 1)
            #     x = (x + 1) / 2
            #     x = x.permute(1, 2, 3, 0).numpy()
            #     x = (255*x).astype(np.uint8)
            #     array_to_video(x, fps=fps, output_file=output_file)
            #     return
            # videos = ae.decode(x)[0]
            # videos = videos.transpose(0, 1)
            # custom_to_video(videos.to(torch.float32), fps=24, output_file='tmp.mp4')
            # import sys;sys.exit()

        # print("input_shape:", input_image.shape, "latent shape:", x.shape)
        current_step_frame = x.shape[2]
        current_step_sp_state = get_sequence_parallel_state()
        if args.sp_size != 1:  # enable sp
            if current_step_frame == 1:  # but image do not need sp
                set_sequence_parallel_state(False)
            else:
                set_sequence_parallel_state(True)
        if get_sequence_parallel_state():
            # 
            x, cond, attn_mask, cond_mask, use_image_num = prepare_parallel_data(x, cond, attn_mask, cond_mask,
                                                                                 args.use_image_num)
            for iter in range(args.train_batch_size * args.sp_size // args.train_sp_batch_size):
                with accelerator.accumulate(model):
                    st_idx = iter * args.train_sp_batch_size
                    ed_idx = (iter + 1) * args.train_sp_batch_size
                    model_kwargs = dict(encoder_hidden_states=cond[st_idx: ed_idx],
                                        attention_mask=attn_mask[st_idx: ed_idx],
                                        encoder_attention_mask=cond_mask[st_idx: ed_idx], use_image_num=use_image_num)
                    run(x[st_idx: ed_idx], model_kwargs, prof_)

        else:
            with accelerator.accumulate(model):
                assert not torch.any(torch.isnan(x)), 'after vae'
                x = x.to(weight_dtype)
                model_kwargs = dict(encoder_hidden_states=cond, attention_mask=attn_mask,
                                    encoder_attention_mask=cond_mask, use_image_num=args.use_image_num)
                run(x, model_kwargs, prof_)
        set_sequence_parallel_state(current_step_sp_state)  # in case the next step use sp, which need broadcast(timesteps)
        if progress_info.global_step >= args.max_train_steps:
            return True
        return False

    def train_all_epoch(prof_=None):
        for epoch in range(first_epoch, args.num_train_epochs):
            progress_info.train_loss = 0.0
            if progress_info.global_step >= args.max_train_steps:
                return True
            if progress_info.local_step > 0:
                print(f'skip first {progress_info.local_step} batches')
                print(f'skip {args.skip_extra} extra batches for sake of stablizing traing' )
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, progress_info.local_step + args.skip_extra)
                for step, data_item in enumerate(skipped_dataloader):
                    if train_one_step(step, data_item, prof_):
                        break
            else:
                for step, data_item in enumerate(train_dataloader):
                    if train_one_step(step, data_item, prof_):
                        break
            progress_info.local_step = 0 #每个epoch结束的时候，local step置为0

    train_all_epoch()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset & dataloader
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data", nargs='+', required=True)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--train_fps", type=int, default=24)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--max_height", type=int, default=320)
    parser.add_argument("--max_width", type=int, default=240)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument('--cfg', type=float, default=0.1)
    parser.add_argument("--dataloader_num_workers", type=int, default=10, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--group_frame", action="store_true")
    parser.add_argument("--group_resolution", action="store_true")

    # text encoder & vae & diffusion model
    parser.add_argument("--model", type=str, choices=list(Diffusion_models.keys()), default="Latte-XL/122")
    parser.add_argument('--enable_8bit_t5', action='store_true')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.125)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument("--compress_kv", action="store_true")
    parser.add_argument("--attention_mode", type=str, choices=['xformers', 'math', 'flash'], default="xformers")
    parser.add_argument('--use_rope', action='store_true')
    parser.add_argument('--compress_kv_factor', type=int, default=1)
    parser.add_argument('--interpolation_scale_h', type=float, default=1.0)
    parser.add_argument('--interpolation_scale_w', type=float, default=1.0)
    parser.add_argument('--interpolation_scale_t', type=float, default=1.0)
    parser.add_argument("--downsampler", type=str, default=None)
    parser.add_argument("--ae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--ae_path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument('--enable_stable_fp32', action='store_true')
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")

    # diffusion setting
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--noise_offset", type=float, default=0.02, help="The scale of noise offset.")
    parser.add_argument("--prediction_type", type=str, default=None, help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")

    # validation & logs
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    parser.add_argument('--guidance_scale', type=float, default=4.5)
    parser.add_argument("--enable_tracker", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=10, help=("Max number of checkpoints to store."))
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                        help=(
                            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
                            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
                            " training using `--resume_from_checkpoint`."
                        ),
                        )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help=(
                            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
                            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
                        ),
                        )
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help=(
                            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
                        ),
                        )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    # optimizer & scheduler
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimizer", type=str, default="adamW", help='The optimizer type to use. Choose between ["AdamW", "prodigy"]')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-02, help="Weight decay to use for unet params")
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=None, help="Weight decay to use for text_encoder")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer and Prodigy optimizers.")
    parser.add_argument("--prodigy_use_bias_correction", type=bool, default=True, help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW")
    parser.add_argument("--prodigy_safeguard_warmup", type=bool, default=True, help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. Ignored if optimizer is adamW")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_beta3", type=float, default=None,
                        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
                             "uses the value of square root of beta2. Ignored if optimizer is adamW",
                        )
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
                        ),
                        )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument("--train_sp_batch_size", type=int, default=1, help="Batch size for sequence parallel training")
    parser.add_argument("--start_from_middle", action='store_true')
    parser.add_argument("--skip_extra", type=int, default=0, help="for stablize training, skip some extra batches")
    parser.add_argument("--reflow", action='store_true')
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--diffusion_formula",
        default = 'DDPM'
    )
    parser.add_argument("--stages", default=3, type=int)

    parser.add_argument("--compile", action='store_true')
    args = parser.parse_args()
    main(args)

import os
import torch
import math
import logging
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from packaging import version
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed

import diffusers
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available


from opensora.adaptor.modules import replace_with_fp32_forwards
from opensora.utils.parallel_states import initialize_sequence_parallel_state, \
    destroy_sequence_parallel_group, get_sequence_parallel_state, set_sequence_parallel_state
from opensora.utils.communications import prepare_parallel_data, broadcast
from opensora.models.diffusion import Diffusion_models, Diffusion_models_class

check_min_version("0.24.0")
logger = get_logger(__name__)

class ProgressInfo:
    def __init__(self, global_step, local_step=0, train_loss=0.0):
        self.global_step = global_step
        self.local_step = local_step
        self.train_loss = train_loss

class AcceleratorTrainer():
    def __init__(self, args):
        '''
        configure basic DDP & logging environ
        '''
        logging_dir = Path(args.output_dir, args.logging_dir)

        if args.enable_stable_fp32:
            replace_with_fp32_forwards()
        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
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
        if args.seed is not None:
            set_seed(args.seed)

        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                os.makedirs(logging_dir, exist_ok=True)
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        
        self.args = args
        self.weight_dtype = weight_dtype
        self.accelerator = accelerator

    def prepare_for_training(self, t2v_models, train_dataloader):
        '''
        move models to device, configure ema, optimizer, lr_scheduler etc. hopefully these will not change
        #TODO: configure lr warm-up
        '''
        args = self.args
        t2v_models.ae.vae.to(self.accelarator.device, dtype = torch.float32)
        t2v_models.text_enc.to(self.accelerator.device, dtype = self.weight_dtype)

        # ema
        if args.use_ema:
            ema_model = deepcopy(t2v_models.model)
            ema_model = EMAModel(ema_model.parameters(), decay=args.ema_decay, update_after_step=args.ema_start_step,
                             model_cls=Diffusion_models_class[args.model], model_config=ema_model.config)
        
        # optimizer
        params_to_optimize = t2v_models.model.parameters()
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

        # lr_scheduler
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        # prepare everything
        logger.info(f'before accelerator.prepare')
        model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            t2v_models.model, optimizer, train_dataloader, lr_scheduler
        )
        logger.info(f'after accelerator.prepare')
        if args.use_ema:
            ema_model.to(self.accelerator.device)
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        self.ema_model = ema_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model = model
        self.train_dataloader = train_dataloader
        self.args = args
    
    def start_training(self, t2v_models, diffusion):
        '''
        start model training, potentially resume from a previous checkpoint
        '''
        args = self.args
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(os.path.basename(args.output_dir), config=vars(args))
        total_batch_size = args.train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps
        total_batch_size = total_batch_size // args.sp_size * args.train_sp_batch_size
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
                self.accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(args.output_dir, path))
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
            disable=not self.accelerator.is_local_main_process,
        )
        progress_info = ProgressInfo(global_step, local_step, train_loss=0.0)

        def sync_gradients_info(loss):
            # save ckpts and print loss info
            # Checks if the accelerator has performed an optimization step behind the scenes
            if args.use_ema:
                ema_model.step(model.parameters())
            progress_bar.update(1)
            progress_info.global_step += 1
            progress_info.local_step += 1
            end_time = time.time()
            one_step_duration = end_time - start_time
            self.accelerator.log({"train_loss": progress_info.train_loss}, step=progress_info.global_step)
            progress_info.train_loss = 0.0
            if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                if progress_info.global_step % args.checkpointing_steps == 0:
                    try:
                        if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{progress_info.global_step}-{progress_info.local_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    except Exception as e:
                        print("attempted to save ckpts, caught error:", e)
                        print('wait until next turn')
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

        def run(model_input, model_kwargs, prof):
            '''
            calculate loss and prefrom optmization
            '''
            global start_time
            start_time = time.time()
            loss = diffusion.get_loss(model, model_input, model_kwargs, args)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = model.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            progress_info.train_loss += avg_loss.detach().item() / args.gradient_accumulation_steps
            if accelerator.sync_gradients:
                sync_gradients_info(loss)
            if accelerator.is_main_process:
                if progress_info.global_step % args.checkpointing_steps == 0:
                    if args.enable_tracker:
                        log_validation(args, model, ae, text_enc.text_enc, train_dataset.tokenizer, accelerator,
                                        weight_dtype, progress_info.global_step)
            if prof is not None:
                prof.step()
            return loss

        def train_one_step(step_, data_item_, prof_=None):
            '''
            shard training batch, vae encode, basically prepare model input for optimization
            '''
            train_loss = 0.0
            x, attn_mask, input_ids, cond_mask = data_item_
            if args.group_frame or args.group_resolution:
                if not args.group_frame:
                    each_latent_frame = torch.any(attn_mask.flatten(-2), dim=-1).int().sum(-1).tolist()
                    print(f'rank: {accelerator.process_index}, step {step_}, special batch has attention_mask '
                            f'each_latent_frame: {each_latent_frame}')
            assert not torch.any(torch.isnan(x)), 'torch.any(torch.isnan(x))'
            x = x.to(accelerator.device, dtype=ae.vae.dtype)  # B C T+num_images H W, 16 + 4
            attn_mask = attn_mask.to(accelerator.device)  # B T+num_images H W
            input_ids = input_ids.to(accelerator.device)  # B 1+num_images L
            cond_mask = cond_mask.to(accelerator.device)  # B 1+num_images L
            input_image = x.clone()
            with torch.no_grad():
                B, N, L = input_ids.shape
                input_ids_ = input_ids.reshape(-1, L)
                cond_mask_ = cond_mask.reshape(-1, L)
                cond = text_enc(input_ids_, cond_mask_)  # B 1+num_images L D
                cond = cond.reshape(B, N, L, -1)
                # Map input images to latent space + normalize latents
                x = ae.encode(x)  # B C T H W
            current_step_frame = x.shape[2]
            current_step_sp_state = get_sequence_parallel_state()
            if args.sp_size != 1:  # ena
                if current_step_frame == 1:  # but image do not need sp
                    set_sequence_parallel_state(False)
                else:
                    set_sequence_parallel_state(True)
            if get_sequence_parallel_state():
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
                    skipped_dataloader = accelerator.skip_first_batches(train_dataloader, progress_info.local_step)
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

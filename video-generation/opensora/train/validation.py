import os
import torch
import pandas as pd
from tqdm import tqdm
from einops import rearrange
from opensora.utils.utils import get_condition
from copy import deepcopy

class PFValidationRunner():
    def __init__(self, evaluation_steps, max_num_samples):
        self.evaluation_steps = evaluation_steps
        self.max_num_samples = max_num_samples

    @torch.no_grad()
    def run_valid(self, model, vae, text_enc, diffusion, valid_dataloader, accelerator, weight_dtype, args, logger):
        valid_loss_list = []
        for step, data_item_ in tqdm(enumerate(valid_dataloader), disable=not accelerator.is_local_main_process):
            x, caption = data_item_
            assert not torch.any(torch.isnan(x)), 'torch.any(torch.isnan(x))'
            x = x.to(accelerator.device, dtype=weight_dtype)
            if not args.load_cached_latents:
                caption = [caption] if isinstance(caption, str) else caption
                cond, cond_mask, pooled_projections = text_enc(caption, device = accelerator.device)
                x = x.to(accelerator.device, dtype=vae.dtype)
                x = vae.encode(x).latent_dist.sample() #already been scaled
            else:
                cond, cond_mask, pooled_projections = caption
                x = x

            assert not torch.any(torch.isnan(x)), 'after vae'
            x = x.to(weight_dtype)
            model_kwargs = dict(encoder_hidden_states=cond,
                                    encoder_attention_mask=cond_mask, use_image_num=args.use_image_num, encoder_pooled_projections = pooled_projections)
            for stage_index in [0, 1, 2]:
                loss = diffusion.get_valid_loss(model, x, model_kwargs, args, stage_index = stage_index, evaluate_steps=self.evaluation_steps)
                valid_loss_list.append(loss)
            if step >= self.max_num_samples:
                break
        stage_0_valid_loss = torch.stack(valid_loss_list[0::3])
        stage_1_valid_loss = torch.stack(valid_loss_list[1::3])
        stage_2_valid_loss = torch.stack(valid_loss_list[2::3])

        stage_0_valid_loss = rearrange(stage_0_valid_loss, "b t s d -> s (b t d)").mean(1).unsqueeze(0)
        stage_1_valid_loss = rearrange(stage_1_valid_loss, "b t s d -> s (b t d)").mean(1).unsqueeze(0)
        stage_2_valid_loss = rearrange(stage_2_valid_loss, "b t s d -> s (b t d)").mean(1).unsqueeze(0)

        stage_0_valid_loss = accelerator.gather(stage_0_valid_loss).mean(0)
        stage_1_valid_loss = accelerator.gather(stage_1_valid_loss).mean(0)
        stage_2_valid_loss = accelerator.gather(stage_2_valid_loss).mean(0)

        if accelerator.is_local_main_process:
            logger.info(f"==================> validation done")
            logger.info(f"======> stage 0 at 7 ar length: {stage_0_valid_loss}")
            logger.info(f"======> stage 1 at 7 ar length: {stage_1_valid_loss}")
            logger.info(f"======> stage 2 at 7 ar length: {stage_2_valid_loss}")
            if accelerator.is_main_process:
                with open(os.path.join(args.output_dir, "stage_0_valid_loss"), 'a') as f:
                    f.write(f"======> stage 0 at 7 ar length: {stage_0_valid_loss}\n")
                with open(os.path.join(args.output_dir, "stage_1_valid_loss"), 'a') as f:
                    f.write(f"======> stage 1 at 7 ar length: {stage_1_valid_loss}\n")
                with open(os.path.join(args.output_dir, "stage_2_valid_loss"), 'a') as f:
                    f.write(f"======> stage 2 at 7 ar length: {stage_2_valid_loss}\n")

class CogvideoValidationRunner():
    def __init__(self, evaluation_steps, max_num_samples):
        self.evaluation_steps = evaluation_steps
        self.max_num_samples = max_num_samples

    @torch.no_grad()
    def run_valid(self, model, vae, text_enc, diffusion, valid_dataloader, accelerator, weight_dtype, args, logger):
        valid_loss_list = []
        for step, data_item_ in tqdm(enumerate(valid_dataloader), disable=not accelerator.is_local_main_process):
            x, caption = data_item_
            assert not torch.any(torch.isnan(x)), 'torch.any(torch.isnan(x))'
            x = x.to(accelerator.device, dtype=weight_dtype)
            if not args.load_cached_latents:
                caption = [caption] if isinstance(caption, str) else caption
                cond, cond_mask, pooled_projections = text_enc(caption, device = accelerator.device)
                x = x.to(accelerator.device, dtype=vae.dtype)
                if args.task == 'i2v':
                    image = x[:, :, :1].clone()
                    x = vae.encode(x) #already been scaled
                    image_latents = vae.encode(image)
                else:
                    x = vae.encode(x) #already been scaled
            else:
                cond, cond_mask, pooled_projections = caption
                x = x
                if args.task == 'i2v': #TODO: currently there is no noise and dropout
                    image_latents = x[:, :, :1].clone()
                    
            assert not torch.any(torch.isnan(x)), 'after vae'
            x = x.to(weight_dtype)
            if args.task == 'i2v':
                image_latents = image_latents.to(weight_dtype)
                x = (x, image_latents)
            model_kwargs = dict(encoder_hidden_states=cond,
                                    encoder_attention_mask=cond_mask, use_image_num=args.use_image_num, encoder_pooled_projections = pooled_projections)
            
            loss = diffusion.get_valid_loss(model, x, model_kwargs, args, evaluate_steps=self.evaluation_steps)
            valid_loss_list.append(loss)
            if step >= self.max_num_samples:
                break
        
        valid_loss = torch.stack(valid_loss_list)
        valid_loss = accelerator.gather(valid_loss).mean()
        if accelerator.is_local_main_process:
            logger.info(f"==================> validation done")
            logger.info(f"======> valid loss: {valid_loss}")
            if accelerator.is_main_process:
                with open(os.path.join(args.output_dir, "valid_loss"), 'a') as f:
                    f.write(f"======> valid loss: {valid_loss}\n")

    @torch.no_grad()
    def run_valid_controlnet(self, model, vae, text_enc, controlnet, cotracker, diffusion, valid_dataloader, accelerator, weight_dtype, args, logger):
        valid_loss_list = []
        for step, data_item_ in tqdm(enumerate(valid_dataloader), disable=not accelerator.is_local_main_process):
            x, caption, batch_embed_scale, condition = data_item_
            _x = deepcopy(x).to(accelerator.device)
            _args = deepcopy(args)
            _args.sample_type = "sparse"
            if condition:
                condition = condition.to(accelerator.device, dtype=weight_dtype)
            else:
                condition = []
                for i in range(_x.shape[0]):
                    _ = get_condition(_x[i], cotracker, accelerator.device, _args) # n t h w
                    condition.append(torch.from_numpy(_))
                condition = torch.stack(condition, dim=0).to(accelerator.device, weight_dtype)

            assert not torch.any(torch.isnan(x)), 'torch.any(torch.isnan(x))'
            x = x.to(accelerator.device, dtype=weight_dtype)
            if not args.load_cached_latents:
                caption = [caption] if isinstance(caption, str) else caption
                cond, cond_mask, pooled_projections = text_enc(caption, device = accelerator.device)
                x = x.to(accelerator.device, dtype=vae.dtype)
                if args.task == 'i2v':
                    image = x[:, :, :1].clone()
                    x = vae.encode(x) #already been scaled
                    image_latents = vae.encode(image)
                else:
                    x = vae.encode(x) #already been scaled
            else:
                cond, cond_mask, pooled_projections = caption
                x = x
                if args.task == 'i2v': #TODO: currently there is no noise and dropout
                    image_latents = x[:, :, :1].clone()
                    
            assert not torch.any(torch.isnan(x)), 'after vae'
            x = x.to(weight_dtype)
            if args.task == 'i2v':
                image_latents = image_latents.to(weight_dtype)
                x = (x, image_latents)
            model_kwargs = dict(encoder_hidden_states=cond,
                                    encoder_attention_mask=cond_mask, use_image_num=args.use_image_num, encoder_pooled_projections = pooled_projections,
                                    weight_dtype = weight_dtype)
            
            loss = diffusion.get_valid_loss_controlnet(condition, controlnet, model, x, model_kwargs, args, evaluate_steps=self.evaluation_steps)
            valid_loss_list.append(loss)
            if step >= self.max_num_samples:
                break
        
        valid_loss = torch.stack(valid_loss_list)
        valid_loss = accelerator.gather(valid_loss).mean()
        if accelerator.is_local_main_process:
            logger.info(f"==================> validation done")
            logger.info(f"======> valid loss: {valid_loss}")
            if accelerator.is_main_process:
                with open(os.path.join(args.output_dir, "valid_loss"), 'a') as f:
                    f.write(f"======> valid loss: {valid_loss}\n")

def get_validation_runner(args):
    if args.diffusion_formula == 'AutoregressivePyramidFlow':
        return PFValidationRunner(evaluation_steps = args.evaluation_steps, max_num_samples=args.max_num_evaluate_samples)
    elif args.diffusion_formula == 'CogVideo':
        return CogvideoValidationRunner(evaluation_steps = args.evaluation_steps, max_num_samples=args.max_num_evaluate_samples)
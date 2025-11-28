import math
import os
import torch
import argparse
import torchvision
import torch.distributed as dist
import numpy as np
from PIL import Image

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler,
                                  FlowMatchEulerDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder, Transformer2DModel
from omegaconf import OmegaConf
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, MT5EncoderModel
from opensora.pyramid_dit.modeling_text_encoder import SD3TextEncoderWithMask
from opensora.pyramid_dit.modeling_pyramid_mmdit import PyramidDiffusionMMDiT
from opensora.video_vae.modeling_causal_vae import CausalVideoVAE

import os, sys

from opensora.adaptor.modules import replace_with_fp32_forwards
from opensora.models.causalvideovae import ae_stride_config, ae_channel_config, ae_norm, ae_denorm, CausalVAEModelWrapper
from opensora.models.diffusion.udit.modeling_udit import UDiTT2V
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V

from opensora.models.text_encoder import get_text_enc
from opensora.utils.utils import save_video_grid

from opensora.sample.pipeline_opensora_sp import OpenSoraPipeline
from opensora.sample.pipeline_opensora_pf import PFOpenSoraPipeline
from opensora.sample.pipeline_arpf import ARPFPipeline
from opensora.sample.flow_matching_scheduler import PyramidFlowMatchEulerDiscreteScheduler
from opensora.sample.pyramid_flow_matching_scheduler import AutoRegressivePyramidFlowMatchEulerDiscreteScheduler

from opensora.flux_modules import (
    PyramidFluxTransformer,
    FluxTextEncoderWithMask,
)

import imageio

try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import initialize_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
    pass
import time



def load_t2v_checkpoint(model_path, use_pyramid_flow = False):
    if args.model_type == 'arpf':
        # TODO: 优化
        # model_path = os.path.join(model_path, "diffusion_transformer_384p")
        model_path = "outputs/25node_stage_a_7_steps_history_400_hqset/checkpoint-3592-3592/model_ema"
        transformer_model = PyramidDiffusionMMDiT.from_pretrained(model_path, cache_dir=args.cache_dir,
                                                        low_cpu_mem_usage=True, device_map=None,
                                                        torch_dtype=weight_dtype, use_flash_attn=False, use_t5_mask=True, add_temp_pos_embed=True, temp_pos_embed_type='rope', use_temporal_causal=True, interp_condition_pos=True)
    elif args.model_type == 'flux_arpf':
        # model_path = os.path.join(model_path, "diffusion_transformer_384p")
        model_path = "outputs/pf_768p/checkpoint-6000-6000/model_ema"
        transformer_model = PyramidFluxTransformer.from_pretrained(
            model_path, torch_dtype=weight_dtype, 
            use_flash_attn=False, use_temporal_causal=True,
            interp_condition_pos=True, axes_dims_rope=[16, 24, 24],
        )
    elif args.model_type == 'udit':
        transformer_model = UDiTT2V.from_pretrained(model_path, cache_dir=args.cache_dir,
                                                        low_cpu_mem_usage=True, device_map=None,
                                                        torch_dtype=weight_dtype)
    elif args.model_type == 'dit':
        transformer_model = OpenSoraT2V.from_pretrained(model_path, cache_dir=args.cache_dir,
                                                        low_cpu_mem_usage=True, device_map=None,
                                                        torch_dtype=weight_dtype)
    else:
        transformer_model = LatteT2V.from_pretrained(model_path, cache_dir=args.cache_dir, low_cpu_mem_usage=False,
                                                     device_map=None, torch_dtype=weight_dtype)
    print('load_t2v_checkpoint, ', model_path)
    # set eval mode
    transformer_model.eval()
    if args.model_type == 'arpf' or args.model_type == 'flux_arpf':
        pipeline = ARPFPipeline(
            vae = vae,
            tokenizer = None,
            text_encoder = text_encoder,
            scheduler = scheduler,
            transformer = transformer_model,
            model_name = args.model_type
        ).to(device)
    elif use_pyramid_flow:
        pipeline = PFOpenSoraPipeline(vae=vae,
                                    text_encoder=text_encoder,
                                    tokenizer=tokenizer,
                                    scheduler=scheduler,
                                    transformer=transformer_model).to(device)
    else:
        pipeline = OpenSoraPipeline(vae=vae,
                                    text_encoder=text_encoder,
                                    tokenizer=tokenizer,
                                    scheduler=scheduler,
                                    transformer=transformer_model).to(device)

    return pipeline

def define_models_opensora(args, weight_dtype, device):
    vae = CausalVAEModelWrapper(args.ae_path)
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        vae.vae.tile_sample_min_size = 512
        vae.vae.tile_latent_min_size = 64
        vae.vae.tile_sample_min_size_t = 29
        vae.vae.tile_latent_min_size_t = 8
        if args.save_memory:
            vae.vae.tile_sample_min_size = 256
            vae.vae.tile_latent_min_size = 32
            vae.vae.tile_sample_min_size_t = 29
            vae.vae.tile_latent_min_size_t = 8
    
    vae.vae_scale_factor = ae_stride_config[args.ae]
    
    text_encoder = MT5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir,
                                                  low_cpu_mem_usage=True, torch_dtype=weight_dtype).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)

    vae.eval()
    text_encoder.eval()

    return vae, text_encoder, tokenizer

def define_models_arpf(args, weight_dtype, device):
    torch_dtype = weight_dtype
    if args.model_type == 'flux_arpf':
        text_encoder = FluxTextEncoderWithMask(args.model_path, torch_dtype=torch.bfloat16)
    else:
        text_encoder = SD3TextEncoderWithMask(args.model_path, torch_dtype=torch.bfloat16)
    # TODO：考虑VAE部分用全精度？
    vae = CausalVideoVAE.from_pretrained(os.path.join(args.model_path, 'causal_video_vae'), torch_dtype=torch_dtype, interpolate=False)
    vae.enable_tiling()
    vae.eval()
    text_encoder.eval()
    return vae, text_encoder

def get_latest_path():
    # Get the most recent checkpoint
    dirs = os.listdir(args.model_path)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None

    return path


def run_model_and_save_images(pipeline, model_path, generator, stages, num_inference_steps_per_stage, init_image):
    video_grids = []
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        args.text_prompt = [i.strip() for i in text_prompt]

    if init_image is not None:
        if init_image.endswith('txt'):
            init_image = open(init_image, 'r').readlines()
            init_image = [i.strip() for i in init_image]
        else:
            init_image = [init_image]

    checkpoint_name = f"{os.path.basename(model_path)}"
    # positive_prompt = "{}"
    
    # negative_prompt = ""
    positive_prompt = "{}, hyper quality, Ultra HD, 8K"
    
    negative_prompt = "cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror"

    for index, prompt in enumerate(args.text_prompt):
        # if index % world_size != local_rank:
        #     continue
        # print('Processing the ({}) prompt, device ({})'.format(prompt, device))
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            if init_image is None:
                if args.no_ar:
                    videos = pipeline.generate_non_ar(positive_prompt.format(prompt),
                                negative_prompt=negative_prompt, 
                                num_frames=args.num_frames,
                                height=args.height,
                                width=args.width,
                                num_inference_steps=args.num_sampling_steps,
                                guidance_scale=args.guidance_scale,
                                video_guidance_scale = args.video_guidance_scale, # TODO: 暂时都是一样的
                                num_images_per_prompt=1,
                                mask_feature=True,
                                device=args.device,
                                max_sequence_length=args.max_sequence_length,
                                generator=generator,
                                stages = stages,
                                num_inference_steps_per_stage = [20, 20, 20],
                                video_num_inference_steps_per_stage = [10, 10, 10],
                                frame_per_unit = args.frames_per_unit,
                                max_history_len = args.max_history_len
                                ).images
                else:
                    videos = pipeline(positive_prompt.format(prompt),
                                    negative_prompt=negative_prompt, 
                                    num_frames=args.num_frames,
                                    height=args.height,
                                    width=args.width,
                                    num_inference_steps=args.num_sampling_steps,
                                    guidance_scale=args.guidance_scale,
                                    video_guidance_scale = args.video_guidance_scale, # TODO: 暂时都是一样的
                                    num_images_per_prompt=1,
                                    mask_feature=True,
                                    device=args.device,
                                    max_sequence_length=args.max_sequence_length,
                                    generator=generator,
                                    stages = stages,
                                    num_inference_steps_per_stage = [20, 20, 20],
                                    video_num_inference_steps_per_stage = [10, 10, 10],
                                    frame_per_unit = args.frames_per_unit,
                                    max_history_len = args.max_history_len
                                    ).images
            else:
                image_idx = min(len(init_image)-1, index)
                init_image_idx = Image.open(init_image[image_idx]).convert('RGB')
                videos = pipeline.generate_i2v(
                    positive_prompt.format(prompt),
                    negative_prompt=negative_prompt, 
                    input_image = init_image_idx,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_sampling_steps,
                    guidance_scale=args.guidance_scale,
                    video_guidance_scale = args.video_guidance_scale, # TODO: 暂时都是一样的
                    num_images_per_prompt=1,
                    mask_feature=True,
                    device=args.device,
                    max_sequence_length=args.max_sequence_length,
                    generator = generator,
                    stages = stages,
                    num_inference_steps_per_stage = [20, 20, 20],
                    video_num_inference_steps_per_stage = [10, 10, 10],
                    frame_per_unit = args.frames_per_unit,
                    max_history_len = args.max_history_len
                ).images
            print(videos.shape)
        
        if nccl_info.rank <= 0:
            try:
                if args.num_frames == 1:
                    videos = videos[:, 0].permute(0, 3, 1, 2)  # b t h w c -> b c h w
                    save_image(videos / 255.0, os.path.join(args.save_img_path,
                                                            f'{args.sample_method}_{index}_{checkpoint_name}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'),
                            nrow=1, normalize=True, value_range=(0, 1))  # t c h w

                else:
                    imageio.mimwrite(
                        os.path.join(
                            args.save_img_path,
                            f'{args.sample_method}_{index}_{checkpoint_name}__gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'
                        ), videos[0],
                        fps=args.fps, quality=6, codec='libx264',
                        output_params=['-threads', '20'])  # highest quality is 10, lowest is 0
            except Exception as e:
                print(e)
                print('Error when saving {}'.format(prompt))
            video_grids.append(videos)
    if nccl_info.rank <= 0:
        video_grids = torch.cat(video_grids, dim=0)

        def get_file_name():
            return os.path.join(args.save_img_path,
                                f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}_{checkpoint_name}.{ext}')

        if args.num_frames == 1:
            save_image(video_grids / 255.0, get_file_name(),
                    nrow=math.ceil(math.sqrt(len(video_grids))), normalize=True, value_range=(0, 1))
        else:
            video_grids = save_video_grid(video_grids)
            imageio.mimwrite(get_file_name(), video_grids, fps=args.fps, quality=6)

        print('save path {}'.format(args.save_img_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default=None, choices=[None, '65x512x512', '65x256x256', '17x256x256'])
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--video_guidance_scale", type=float, default=5.0)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--model_type', type=str, default="dit", choices=['dit', 'udit', 'latte', 'arpf', 'flux_arpf'])
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--save_memory', action='store_true')
    parser.add_argument("--num_stages", default=3, type=int)
    parser.add_argument("--num_inference_steps_per_stage", default='20,20,20')
    parser.add_argument("--frames_per_unit", type = int, default=1)
    parser.add_argument("--max_history_len", type = int, default=4)
    parser.add_argument("--init_image", type = str, default=None)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--no_ar', action='store_true')
    args = parser.parse_args()

    if torch_npu is not None:
        npu_config.print_msg(args)

    # 初始化分布式环境
    use_pyramid_flow = False
    num_stages = int(args.num_stages)
    num_inference_steps_per_stage = args.num_inference_steps_per_stage.split(',')
    num_inference_steps_per_stage = [int(num) for num in num_inference_steps_per_stage]
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    print('world_size', world_size)
    if torch_npu is not None and npu_config.on_npu:
        torch_npu.npu.set_device(local_rank)
    else:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)
    # if we want to generate images with sp script, we need to set the seed, however for video generation, this should never be set
    if args.seed is not None:
        torch.manual_seed(int(args.seed)) 
        g = torch.torch.Generator()
        g.manual_seed(int(args.seed))
    else:
        g = None
    weight_dtype = torch.bfloat16
    device = torch.cuda.current_device()
   
    if args.model_type == 'arpf' or args.model_type == 'flux_arpf':

        vae, text_encoder = define_models_arpf(args, weight_dtype, device)
    else:
        vae, text_encoder, tokenizer = define_models_opensora(args, weight_dtype, device)

    if args.model_type == 'arpf' or args.model_type == 'flux_arpf':
        print('using auto regressive pyramid flow matching')
        stage_range = np.linspace(0,1, num_stages+1)
        scheduler = AutoRegressivePyramidFlowMatchEulerDiscreteScheduler(
            stages = num_stages,
            stage_range = stage_range,
            gamma = 1/3
        )
    elif args.sample_method == 'DDIM':  #########
        scheduler = DDIMScheduler(clip_sample=False)
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':  #############
        scheduler = DDPMScheduler(clip_sample=False)
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler()
    elif args.sample_method == 'HeunDiscrete':  ########
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == 'KDPM2AncestralDiscrete':  #########
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif args.sample_method == 'flow':
        print('using flow matching')
        scheduler = FlowMatchEulerDiscreteScheduler()
    elif args.sample_method == 'pf-flow':
        print('using pyramid flow matching')
        stage_range = np.linspace(0,1, num_stages+1)
        scheduler = PyramidFlowMatchEulerDiscreteScheduler(
            stages = num_stages,
            stage_range = stage_range,
            gamma = 1/3
        )
        use_pyramid_flow = True

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    if args.num_frames == 1:
        video_length = 1
        ext = 'jpg'
    else:
        ext = 'mp4'

    latest_path = None
    save_img_path = args.save_img_path
    # while True:
        # cur_path = get_latest_path()
        # print(cur_path, latest_path)
        # if cur_path == latest_path:
        #     time.sleep(5)
        #     continue
        # time.sleep(1)
        # latest_path = cur_path

        # if npu_config is not None:
        #     npu_config.print_msg(f"The latest_path is {latest_path}")
        # else:
        #     print(f"The latest_path is {latest_path}")


    if latest_path is None:
        latest_path = ''
    # full_path = f"{args.model_path}/{latest_path}/model_ema"
    full_path = f"{args.model_path}"
    # full_path = f"{args.model_path}/{latest_path}/model"
    pipeline = load_t2v_checkpoint(full_path, use_pyramid_flow)
    print('load model')
    run_model_and_save_images(pipeline, latest_path, generator=g, stages = num_stages, num_inference_steps_per_stage=num_inference_steps_per_stage, init_image = args.init_image)

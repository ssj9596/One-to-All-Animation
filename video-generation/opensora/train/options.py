import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    basic_group = parser.add_argument_group("basic_group", 'arguments for basic training')
    basic_group.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    basic_group.add_argument("--output_dir", type=str, default=None, help="The output directory where the model predictions and checkpoints will be written.")
    basic_group.add_argument("--checkpointing_steps", type=int, default=500)
    basic_group.add_argument("--checkpoints_total_limit", type=int, default=10, help=("Max number of checkpoints to store."))
    basic_group.add_argument("--resume_from_checkpoint", type=str, default=None)
    basic_group.add_argument("--logging_dir", type=str, default="logs")
    basic_group.add_argument("--report_to", type=str, default="tensorboard")
    basic_group.add_argument("--num_train_epochs", type=int, default=100)
    basic_group.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    basic_group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    basic_group.add_argument("--allow_tf32", action="store_true") 
    basic_group.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    basic_group.add_argument("--start_from_middle", action='store_true')
    basic_group.add_argument('--conditional_clear_ratio', type=float, default=0.5)
    basic_group.add_argument("--skip_extra", type=int, default=0, help="for stablize training, skip some extra batches")

    optimizer_group = parser.add_argument_group('optimizer_group', 'arguments for optimizer')
    optimizer_group.add_argument("--optimizer", type=str, default="adamW", help='The optimizer type to use. Choose between ["AdamW", "prodigy"]')
    optimizer_group.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW")
    optimizer_group.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers.")
    optimizer_group.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers.")
    optimizer_group.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    optimizer_group.add_argument("--adam_weight_decay", type=float, default=1e-02, help="Weight decay to use for unet params")
    optimizer_group.add_argument("--adam_weight_decay_text_encoder", type=float, default=None, help="Weight decay to use for text_encoder")
    optimizer_group.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer and Prodigy optimizers.")
    optimizer_group.add_argument("--prodigy_use_bias_correction", type=bool, default=True, help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW")
    optimizer_group.add_argument("--prodigy_safeguard_warmup", type=bool, default=True, help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. Ignored if optimizer is adamW")
    optimizer_group.add_argument("--prodigy_beta3", type=float, default=None)

    lr_group = parser.add_argument_group('lr_group', 'arguments for learning rate')
    lr_group.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    lr_group.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    lr_group.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    lr_group.add_argument("--lr_scheduler", type=str, default="constant")
    lr_group.add_argument("--lr_num_cycles", type=int, default=1)
    lr_group.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    lr_group.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    diffusion_group = parser.add_argument_group("diffusion_group", 'arguments for diffusion')
    diffusion_group.add_argument("--diffusion_formula", default = 'DDPM')
    diffusion_group.add_argument(
            "--weighting_scheme",
            type=str,
            default="logit_normal",
            choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
        )
    diffusion_group.add_argument("--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme.")
    diffusion_group.add_argument("--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme.")
    diffusion_group.add_argument("--mode_scale", type=float, default=1.29, help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.")
    diffusion_group.add_argument("--stages", default=3, type=int)
    diffusion_group.add_argument("--frame_per_unit", default=1, type=int)
    diffusion_group.add_argument("--max_history_len", default=4, type=int)
    diffusion_group.add_argument("--task", default='t2v')
    diffusion_group.add_argument("--repeat", action="store_true", default=False, help="repeat in i2v")
    diffusion_group.add_argument("--restore_video", action="store_true", default=False, help="restore_video")

    lora_group = parser.add_argument_group("lora_group", 'arguments for lora fine-tuning')
    lora_group.add_argument("--lora", action='store_true')
    lora_group.add_argument(
            "--rank",
            type=int,
            default=128,
            help=("The dimension of the LoRA update matrices."),
        )
    lora_group.add_argument(
            "--lora_alpha",
            type=float,
            default=128,
            help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
        )

    data_group = parser.add_argument_group("data_group", 'arguments for dataset')
    data_group.add_argument("--dataset", type=str, required=True)
    data_group.add_argument("--data", nargs='+', required=True)
    data_group.add_argument("--train_fps", type=int, default=24)
    data_group.add_argument('--cfg', type=float, default=0.1)
    data_group.add_argument('--refimg_cfg', type=float, default=0.0)
    data_group.add_argument('--refimg_crop', type=float, default=0.0)
    data_group.add_argument("--dataloader_num_workers", type=int, default=10, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    data_group.add_argument("--bucket_name", default='default', type=str)
    data_group.add_argument("--one_more_second", action='store_true')
    data_group.add_argument("--ignore_timestamps", action='store_true') 
    data_group.add_argument("--max_unit_num", type=int, default=8)
    data_group.add_argument("--load_condition", action='store_true')
    data_group.add_argument("--face_mask", action='store_true')
    data_group.add_argument("--hand_mask", action='store_true')
    data_group.add_argument("--face_weight", type=float, default=2.0)
    data_group.add_argument("--join_posecfg", action='store_true')
    data_group.add_argument('--pose_cfg', type=float, default=0.0)
    data_group.add_argument('--zoomin', type=float, default=0.0)
    data_group.add_argument("--only_ref", action='store_true')
    data_group.add_argument("--pose_aug", action='store_true')
    
    
    data_group.add_argument('--i2v', type=float, default=0.0)# first frame as ref 
    data_group.add_argument('--token_replace_prob', type=float, default=0.0)
    data_group.add_argument("--pose_poolA", type=str, default="")
    data_group.add_argument("--pose_poolB", type=str, default="")
    data_group.add_argument("--change_pose", type=float, default=0.0)
    data_group.add_argument("--ref_keep", type=float, default=0.0)
    
    
    

    model_group = parser.add_argument_group("model_group", 'arguments for model')
    model_group.add_argument("--pretrained", type=str, default=None)
    model_group.add_argument('--enable_stable_fp32', action='store_true')
    model_group.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    model_group.add_argument("--dit_name")
    # model_group.add_argument("--dit_name", choices=['cogvideo', 'mochi', "hunyuan", "pyramid_flow", "hunyuanref", "hunyuanend2end","hunyuan_refextractor","hunyuan_tokenmerge"])
    model_group.add_argument("--config_path")
    model_group.add_argument("--only_patch_embeding", action='store_true')
    model_group.add_argument("--training_modules", nargs='+', default=[], 
                         help="Names of modules to set requires_grad to True. Separate multiple modules with spaces.")
    

    ema_group = parser.add_argument_group('ema_group', 'arguments for save ema model')
    ema_group.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    ema_group.add_argument("--ema_decay", type=float, default=0.999)
    ema_group.add_argument("--ema_start_step", type=int, default=0)

    encoder_group = parser.add_argument_group("encoder_group", 'argument for text encoder')
    # encoder_group.add_argument("--text_encoder_name", choices=['t5', 'flux', 'hunyuan', "hunyuan_image"])
    encoder_group.add_argument("--text_encoder_name")
    encoder_group.add_argument("--text_encoder_path")
    encoder_group.add_argument("--max_sequence_length", type=int, default=226)

    vae_group = parser.add_argument_group("vae_group", 'argument for vae')
    # vae_group.add_argument("--vae_name", choices=['cogvideo', 'pyramid_flow', 'mochi', 'hunyuan'])
    vae_group.add_argument("--vae_name")
    vae_group.add_argument("--vae_path")

    controlnet_group = parser.add_argument_group('controlnet_group', 'argument for controlnet group')
    controlnet_group.add_argument("--init_from_transformer", action='store_true')
    controlnet_group.add_argument("--frozen_controlnet", action='store_true')
    controlnet_group.add_argument("--controlnet_name", type=str, default='cogvideo-controlnet')
    controlnet_group.add_argument("--controlnet_config_path", type=str, default=None)
    controlnet_group.add_argument("--pretrained_controlnet", type=str, default=None)
    controlnet_group.add_argument("--conditioning_scale", default=-1, type=int)
    


    cache_group = parser.add_argument_group('cache_group', 'argument for loading cache')
    cache_group.add_argument("--load_encoders", action='store_true')
    cache_group.add_argument("--load_cached_latents", action='store_true')
    cache_group.add_argument("--null_embedding_path")
    cache_group.add_argument("--temporal_downscale", type=int, default=4)

    valid_group = parser.add_argument_group("valid_group", 'argument for valid group')
    valid_group.add_argument("--do_valid", action='store_true')
    valid_group.add_argument("--valid_data", nargs='+', required=True)
    valid_group.add_argument("--evaluation_steps", default=5, type=int)
    valid_group.add_argument("--evaluation_every", default=1000, type=int)
    valid_group.add_argument("--max_num_evaluate_samples", default=5, type=int)
    valid_group.add_argument("--extra_sample_steps", default=1, type=int)

    ipadapter_group = parser.add_argument_group("ipadapter_group", 'argument for ip_adapter group')
    ipadapter_group.add_argument("--ipadapter_config_path", type=str, default=None)
    ipadapter_group.add_argument("--image_encoder", type=str, default=None)
    ipadapter_group.add_argument("--refblocks_config_path", type=str, default=None)
    ipadapter_group.add_argument("--refextractor_config_path", type=str, default=None)
    ipadapter_group.add_argument("--frozen_refextractor", action='store_true')

    return parser
import argparse

parser = argparse.ArgumentParser()

# dataset & dataloader
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data", type=str, required='')
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
parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
parser.add_argument("--group_frame", action="store_true")
parser.add_argument("--group_resolution", action="store_true")

# text encoder & vae & diffusion model
parser.add_argument("--model", type=str, default="Latte-XL/122")
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
parser.add_argument("--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store."))
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

args = parser.parse_args([
    "--model", "OpenSoraT2V-ROPE-L/122",
    "--dataset", "t2v",
    "--text_encoder_name", "google/mt5-xxl",
    "--cache_dir", "./cache_dir",
    "--data", "example93frames100.csv",
    "--ae", "CausalVAEModel_D4_4x8x8",
    "--ae_path", "xxx",
    "--sample_rate", "1",
    "--num_frames", "93",
    "--max_height", "480",
    "--max_width", "640",
    "--interpolation_scale_t", "1.0",
    "--interpolation_scale_h", "1.0",
    "--interpolation_scale_w", "1.0",
    "--attention_mode", "xformers",
    "--gradient_checkpointing",
    "--train_batch_size", "1",
    "--dataloader_num_workers", "10",
    "--gradient_accumulation_steps", "1",
    "--max_train_steps", "1000000",
    "--learning_rate", "1e-4",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--mixed_precision", "bf16",
    "--report_to", "wandb",
    "--checkpointing_steps", "250",
    "--allow_tf32",
    "--model_max_length", "512",
    "--use_image_num", "0",
    "--tile_overlap_factor", "0.125",
    "--snr_gamma", "5.0",
    "--use_ema",
    "--ema_start_step", "0",
    "--cfg", "0.1",
    "--noise_offset", "0.02",
    "--use_rope",
    "--resume_from_checkpoint", "latest",
    "--ema_decay", "0.999",
    "--enable_tiling",
    "--speed_factor", "1.0",
    "--group_frame",
    "--sp_size", "8",
    "--train_sp_batch_size", "2",
    "--pretrained", "xxx",
    "--output_dir", "bs16x8x1_93x480p_lr1e-4_snr5_ema999_opensora122_rope_mt5xxl_speed1.0"
])

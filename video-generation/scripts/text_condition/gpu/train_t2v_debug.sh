export WANDB_MODE="offline"
export ENTITY="linbin"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export PDSH_RCMD_TYPE=ssh
# NCCL setting
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# export NCCL_ALGO=Tree

python opensora/train/train_t2v.py\
      --model flux_ckpt \
      --text_encoder_name flux_ckpt \
      --cache_dir "./cache_dir" \
      --dataset t2v_multi_res \
      --data "eating_boating_driving.csv"   \
      --ae flux_ckpt/causal_video_vae \
      --ae_path "flux_ckpt/causal_video_vae" \
      --sample_rate 1 \
      --num_frames 93 \
      --max_height 480 \
      --max_width 640 \
      --interpolation_scale_t 1.0 \
      --interpolation_scale_h 1.0 \
      --interpolation_scale_w 1.0 \
      --attention_mode flash \
      --gradient_checkpointing \
      --train_batch_size=1 \
      --dataloader_num_workers 0 \
      --gradient_accumulation_steps=1 \
      --learning_rate=0 \
      --optimizer adamw \
      --adam_beta1 0.9 \
      --adam_beta2 0.95 \
      --adam_weight_decay 1e-4 \
      --lr_scheduler="constant_with_warmup" \
      --lr_warmup_steps=1000 \
      --mixed_precision="bf16" \
      --report_to=None \
      --checkpointing_steps=100 \
      --allow_tf32 \
      --model_max_length 256 \
      --use_image_num 0 \
      --tile_overlap_factor 0.125 \
      --snr_gamma 5.0 \
      --use_ema \
      --ema_start_step 0 \
      --cfg 0.1 \
      --noise_offset 0.02 \
      --use_rope \
      --resume_from_checkpoint="latest" \
      --ema_decay 0.999 \
      --enable_tiling \
      --speed_factor 1.0 \
      --group_frame \
      --sp_size 1 \
      --train_sp_batch_size 1 \
      --output_dir="outputs/debug_768" \
      --checkpoints_total_limit 3 \
      --seed 42 \
      --num_train_epochs 10 \
      --reflow \
      --diffusion_formula AutoregressivePyramidFlow \
      --model_type flux_arpf \
      --stages 3 \
      --train_fps 24 \
      --evaluation_steps 5 \
      --evaluation_every 100 \
      --valid_data valid.csv \
      --max_num_evaluate_samples 10
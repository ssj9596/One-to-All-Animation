#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD
export NCCL_TIMEOUT=3600000

OUTPUT_DIR="outputs_wanx14b/train14b_posecontrol_prefix_2d_tokenreplace"
mkdir -p ${OUTPUT_DIR}

LOG_FILE="${OUTPUT_DIR}/train14b_posecontrol_prefix_2d_tokenreplace.log"

echo "=========================================="
echo "  Training Wan2.1-14B with Pose Control & Token Replace"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo ""

# Classifier-Free Guidance
# cfg 1.0 -> dropout text condition during training (text="")
# pose_cfg 0.1 -> 10% probability dropout pose control and reference image (black image)
# refimg_cfg 0.1 -> 10% probability dropout reference image (black image)
# refimg_crop 0.3 -> 30% probability mask reference image
# zoomin 0.1 -> 10% probability zoom in
# bucket_name 768_v2 -> Training with 768 resolution (20% 512 / 80% 768 see opensora/utils/bucket.py)
# change_pose 0.7 70% face region enhancement
# ref_keep -> dropout pose control and keep reference
# token_replace_prob
accelerate launch \
      --config_file scripts/accelerate_configs/hk_multinode_config.yaml \
      --num_processes 8 \
      --num_machines 1 \
      opensora/train/wanx_train/train_wanx_refextractor_mask2_controlnet2.py \
      --seed 43 \
      --output_dir="${OUTPUT_DIR}" \
      --checkpointing_steps=1000 \
      --checkpoints_total_limit 2 \
      --report_to tensorboard \
      --num_train_epochs 100 \
      --gradient_accumulation_steps=1 \
      --allow_tf32 \
      --mixed_precision="bf16" \
      --optimizer adam \
      --adam_beta1 0.9 \
      --adam_beta2 0.95 \
      --adam_weight_decay 1e-4 \
      --learning_rate=1e-5 \
      --lr_scheduler="constant_with_warmup" \
      --lr_warmup_steps=5000 \
      --max_grad_norm 1.0 \
      --diffusion_formula FlowMatching \
      --load_condition \
      --dataloader_num_workers 4 \
      --dataset bodydance_3 \
      --data "../datasets/opensource_dataset/combined_imgvid_dataset.csv" \
      --bucket_name 768_v2 \
      --cfg 1.0 \
      --pose_cfg 0.1 \
      --refimg_cfg 0.1 \
      --refimg_crop 0.3 \
      --zoomin 0.1 \
      --change_pose 0.5 \
      --ref_keep 0.3 \
      --one_more_second \
      --train_fps 16 \
      --ema_start_step 0 \
      --ema_decay 0.999 \
      --evaluation_steps 5 \
      --evaluation_every 1000 \
      --valid_data "../datasets/opensource_dataset/combined_imgvid_dataset.csv"  \
      --max_num_evaluate_samples 10 \
      --dit_name wanx_refextractor_2d_controlnet_prefix \
      --config_path configs/wan2.1_t2v_14b.json \
      --text_encoder_name wanx-t2v \
      --text_encoder_path ../pretrained_models/Wan2.1-T2V-14B-Diffusers/ \
      --vae_name wanx \
      --vae_path ../pretrained_models/Wan2.1-T2V-14B-Diffusers/vae/ \
      --refextractor_config_path configs/wan2.1_t2v_14b_refextractor_2d_withmask2.json \
      --controlnet_config_path configs/wan2.1_t2v_14b_controlnet_1.json \
      --load_encoders \
      --task t2v \
      --face_mask \
      --hand_mask \
      --face_weight 1.0 \
      --gradient_checkpointing \
      --training_modules attn1 patch_embedding input_hint_block controlnet image_to_cond \
      --resume_from_checkpoint="latest" \
      --pretrained ./outputs_wanx14b/train14b_posecontrol_prefix_2d/checkpoint-xxx/fp32_model_xxx \
      --pose_poolA ../datasets/opensource_pose_pool/images_pose_pool_24_18pt_abs.csv \
      --pose_poolB ../datasets/opensource_pose_pool/images_pose_pool_18_16pt_abs.csv \
      --init_from_transformer \
      --token_replace_prob 0.2 \
    2>&1 | tee -a "${LOG_FILE}"


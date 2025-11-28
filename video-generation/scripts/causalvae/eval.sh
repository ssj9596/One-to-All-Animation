# REAL_DATASET_DIR=/remote-home1/dataset/OpenMMLab___Kinetics-400/raw/Kinetics-400/videos_val/
REAL_DATASET_DIR=valid_gen/vae_1node_sr1_nf33_res256_subset130/origin
EXP_NAME=decoder
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=256
SUBSET_SIZE=130
METRIC=psnr

python opensora/models/causalvideovae/eval/eval_common_metric.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir valid_gen/vae_1node_sr1_nf33_res256_subset130 \
    --device cuda:0 \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --crop_size ${RESOLUTION} \
    --metric ${METRIC}
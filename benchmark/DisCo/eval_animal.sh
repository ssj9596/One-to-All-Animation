#!/bin/bash
# pip install pytorch_fid ffmpeg-python  imageio
# pip install scikit-image lpips
# pip uninstall ffmpeg -y
# pip install imageio-ffmpeg
pip uninstall ffmpeg-python -y
pip install ffmpeg-python
# export CUDA_VISIBLE_DEVICES=0    #
base_dir="../temp_eval_cartoon_frames"

for dataset in "$base_dir"/*; do
    # 只处理目录
    if [ -d "$dataset" ]; then
        folder_name=$(basename "$dataset")
        update_name="${folder_name}"
        update_path="${base_dir}/${update_name}"
        metrics_file="${update_path}/metrics_fid-vid_fvd_dtssd_16frames.json"


        echo "============== 开始处理任务: $folder_name =============="

        update_folder="$update_path"
        pred_folder="$update_folder/pred"
        gt_folder="$update_folder/gt"


        # Step 2: FID
        for clip_id in $(ls "$pred_folder"); do
            pred_path="$pred_folder/$clip_id"
            gt_path="$gt_folder/$clip_id"

            if [ -d "$pred_path" ] && [ -d "$gt_path" ]; then
                fid_file="$update_folder/${clip_id}_pytorch_fid.txt"
                python -m pytorch_fid "$pred_path" "$gt_path" --device cuda:0 > "$fid_file"
                echo "[INFO] FID 保存到 $fid_file"
            else
                echo "[WARN] $clip_id 缺失 pred 或 gt 目录"
            fi
        done

        # Step 3: L1/SSIM/LPIPS/PSNR
        for clip_id in $(ls "$pred_folder"); do
            pred_path="$pred_folder/$clip_id"
            gt_path="$gt_folder/$clip_id"

            if [ -d "$pred_path" ] && [ -d "$gt_path" ]; then
                metrics_file="$update_folder/${clip_id}_metrics_l1_ssim_lpips_psnr.json"
                python tool/metrics/metric_center.py \
                    --root_dir "$update_folder" \
                    --path_gen "$pred_path" \
                    --path_gt "$gt_path" \
                    --type l1 ssim lpips psnr \
                    --write_metric_to "$metrics_file"
                echo "[INFO] 指标保存到 $metrics_file"
            else
                echo "[WARN] $clip_id 缺失 pred 或 gt 目录"
            fi
        done

        # Step 4: 生成 MP4
        python tool/video/yz_gen_gifs_for_fvd_subfolders.py -i "${pred_folder}" -o "${pred_folder}_16framemp4" --fps 3 --format mp4
        python tool/video/yz_gen_gifs_for_fvd_subfolders.py -i "${gt_folder}" -o "${gt_folder}_16framemp4" --fps 3 --format mp4

        # Step 5: FVD / FID-VID
        python tool/metrics/metric_center.py \
            --root_dir . \
            --path_gen "${pred_folder}_16framemp4" \
            --path_gt "${gt_folder}_16framemp4" \
            --type fid-vid fvd \
            --write_metric_to "${update_folder}/metrics_fid-vid_fvd_dtssd_16frames.json" \
            --number_sample_frames 16 --sample_duration 16

        echo "============== 完成任务: $folder_name =============="
    fi
done

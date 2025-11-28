#!/usr/bin/env python3
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

# 定义跳过的目录名列表
SKIP_VIDEOS = {
    "mmexport1758617134089",
    "mmexport1758617147981",
    "mmexport1758617110699",
    "mmexport1758617160088",
    "mmexport1758617106367",
}

def copy_pred_gt(src_base: Path, dst_base: Path):
    pred_dst = dst_base / "pred"
    gt_dst = dst_base / "gt"
    pred_dst.mkdir(parents=True, exist_ok=True)
    gt_dst.mkdir(parents=True, exist_ok=True)

    # 遍历 MimicMotion 下的所有编号文件夹
    for video_dir in sorted(src_base.iterdir()):
        if not video_dir.is_dir():
            continue

        # 跳过名单中的 video_dir
        if video_dir.name in SKIP_VIDEOS:
            print(f"[SKIP] 跳过视频: {video_dir.name}")
            continue

        pred_src = video_dir / "pred"
        gt_src = video_dir / "gt"

        # 目标 clip 文件夹（只为没跳过的创建）
        pred_out = pred_dst / video_dir.name
        gt_out = gt_dst / video_dir.name
        pred_out.mkdir(parents=True, exist_ok=True)
        gt_out.mkdir(parents=True, exist_ok=True)

        # 复制 pred
        if pred_src.exists():
            for frame in tqdm(sorted(pred_src.iterdir()), desc=f"COPY pred {video_dir.name}", unit="frame"):
                if frame.is_file():
                    shutil.copy2(frame, pred_out / frame.name)

        # 复制 gt
        if gt_src.exists():
            for frame in tqdm(sorted(gt_src.iterdir()), desc=f"COPY gt {video_dir.name}", unit="frame"):
                if frame.is_file():
                    shutil.copy2(frame, gt_out / frame.name)

    print(f"[DONE] 已生成到 {dst_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量复制 MimicMotion 下所有视频的 pred/gt 到新目录(保留原文件名)")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="原始 MimicMotion 根目录")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="新目录 MimicMotion_update")
    args = parser.parse_args()

    copy_pred_gt(Path(args.input_dir), Path(args.output_dir))

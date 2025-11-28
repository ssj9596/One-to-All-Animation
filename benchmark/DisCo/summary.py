import json
from pathlib import Path
import statistics
from tabulate import tabulate  # pip install tabulate

def parse_metrics_dir(model_update_dir: Path):

    l1_vals, ssim_vals, lpips_vals, psnr_vals, fid_vals = [], [], [], [], []
    per_clip_records = []
    fvd_3drn50, fvd_3dinception = None, None

    for file in sorted(model_update_dir.iterdir()):
        if file.name.endswith("_metrics_l1_ssim_lpips_psnr.json"):
            clip_id = file.name.split("_metrics_l1_ssim_lpips_psnr.json")[0]
            with open(file) as f:
                data = json.load(f)

            l1_vals.append(data.get("L1"))
            ssim_vals.append(data.get("SSIM"))
            lpips_vals.append(data.get("LPIPS"))
            psnr_vals.append(data.get("PSNR"))

            fid_file = model_update_dir / f"{clip_id}_pytorch_fid.txt"
            fid_val = None
            if fid_file.exists():
                with open(fid_file) as ff:
                    content = ff.read().strip()
                try:
                    fid_val = float(content.split()[-1])
                    fid_vals.append(fid_val)
                except Exception:
                    pass

            per_clip_records.append([
                clip_id,
                f"{data.get('L1'):.6f}",
                f"{data.get('SSIM'):.6f}",
                f"{data.get('LPIPS'):.6f}",
                f"{data.get('PSNR'):.6f}",
                f"{fid_val:.6f}" if fid_val is not None else "N/A"
            ])

        elif file.name == "metrics_fid-vid_fvd_dtssd_16frames.json":
            with open(file) as f:
                data = json.load(f)
            fvd_3drn50 = data.get("FVD-3DRN50")
            fvd_3dinception = data.get("FVD-3DInception")

    metrics_summary = [
        statistics.mean(l1_vals) if l1_vals else None,
        statistics.mean(ssim_vals) if ssim_vals else None,
        statistics.mean(lpips_vals) if lpips_vals else None,
        statistics.mean(psnr_vals) if psnr_vals else None,
        statistics.mean(fid_vals) if fid_vals else None,
        fvd_3drn50,
        fvd_3dinception
    ]

    return metrics_summary, per_clip_records

def main():
    base_dir = Path("../temp_eval_tiktok_frames")
    # base_dir = Path("../temp_eval_cartoon_frames")
    model_summary_all = []
    clip_detail_all = []

    for model_name in sorted(base_dir.iterdir()):
        if model_name.is_dir():
            metrics_summary, per_clip_records = parse_metrics_dir(model_name)

            model_summary_all.append(
                [model_name] + [f"{v:.6f}" if v is not None else "N/A" for v in metrics_summary]
            )

            for clip_row in per_clip_records:
                clip_detail_all.append([model_name] + clip_row)

    summary_header = ["Model", "L1_avg", "SSIM_avg", "LPIPS_avg", "PSNR_avg",
                      "FID_avg", "FVD-3DRN50", "FVD-3DInception"]
    detail_header = ["Model", "VideoID", "L1", "SSIM", "LPIPS", "PSNR", "FID"]


    summary_table = tabulate(model_summary_all, headers=summary_header, tablefmt="grid")
    detail_table = tabulate(clip_detail_all, headers=detail_header, tablefmt="grid")


    (base_dir / "model_summary.txt").write_text(summary_table)
    (base_dir / "clip_details.txt").write_text(detail_table)

    print("\n===== 模型总体指标 =====")
    print(summary_table)
    print("\n===== 每视频详细指标 =====")
    print(detail_table)
    print(f"\n[DONE] 已保存到: {base_dir}/model_summary.txt 和 clip_details.txt")

if __name__ == "__main__":
    main()

<div align="center">

<h1>
    <br> 
    One-to-All Animation: Alignment-Free  <br> Character Animation 
    and Image Pose Transfer
</h1>

<p>
Shijun Shi<sup>1*</sup>, Jing Xu<sup>2*</sup>, Zhihang Li<sup>3</sup>, Chunli Peng<sup>4</sup>, Xiaoda Yang<sup>5</sup>, Lijing Lu<sup>3</sup>,  <br> Kai Hu<sup>1†</sup>, Jiangning Zhang<sup>5†</sup>
</p>

<p style="font-size: 0.9em; color: #666;">
<sup>1</sup>Jiangnan University &nbsp;
<sup>2</sup>University of Science and Technology of China &nbsp;
<sup>3</sup>Chinese Academy of Sciences<br>
<sup>4</sup>Beijing University of Posts and Telecommunications &nbsp;
<sup>5</sup>Zhejiang University
</p>

<p style="font-size: 0.85em; color: #888;">
<sup>*</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding authors
</p>


<p align="center">
  <a href="https://ssj9596.github.io/one-to-all-animation-project/" target="_blank">
    <img src="https://img.shields.io/badge/🌐%20Project%20Page-Visit%20Website-4285F4?style=for-the-badge&logoColor=white" height="30"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2511.22940" target="_blank">
    <img src="https://img.shields.io/badge/📄%20arXiv-2511.22940-B31B1B?style=for-the-badge&logoColor=white" height="30"/>
  </a>
</p>
</div>

<br>

## 🌟 Highlights

We provide a **complete and reproducible** training and evaluation pipeline:

- ✅ **Full Training Code**: Three-stage progressive training from scratch
- ✅ **Complete Benchmarks**: Reproduction code and pre-trained checkpoints
- ✅ **Flexible Training Codebase**: Multi-resolution, multi-aspect-ratio, and multi-frame training codebase
- ✅ **Datasets**: Pre-processed open-source datasets + self-collected cartoon data

<br>

## 🔥 Update
- [2025.11] Paper reproduction and evaluation code released.
- [2025.11] [Sample training data and Benchmark](https://huggingface.co/datasets/MochunniaN1/One-to-All-sub) on HuggingFace released.
- [2025.11] Inference and Training codes are released.
- [2025.11] [1.3B-v1](https://huggingface.co/MochunniaN1/One-to-All-1.3b_1), [1.3B-v2](https://huggingface.co/MochunniaN1/One-to-All-1.3b_2) and  [14B](https://huggingface.co/MochunniaN1/One-to-All-14b) checkpoints are released.

<br>

## 🎭 Showcase

Our model can adapt a single reference image to various motion patterns, demonstrating flexible motion control capabilities.

#### 14B Model

<table align="center">
  <tr>
    <th style="text-align: center;">Reference</th>
    <th style="text-align: center;">Motion 1</th>
    <th style="text-align: center;">Motion 2</th>
    <th style="text-align: center;">Motion 3</th>
  </tr>
  <tr>
    <td align="center" style="padding: 2px;"><img src="./examples/new_examples/1.png" height="250"/></td>
    <td align="center" style="padding: 2px;"><img src="assets/14b_examples/ref_1_motion1.gif" height="250"/></td>
    <td align="center" style="padding: 2px;"><img src="assets/14b_examples/ref_1_motion2.gif" height="250"/></td>
    <td align="center" style="padding: 2px;"><img src="assets/14b_examples/ref_1_motion3.gif" height="250"/></td>
  </tr>
  <tr>
    <td align="center" style="padding: 2px;"><img src="./examples/new_examples/2.png" height="250"/></td>
    <td align="center" style="padding: 2px;"><img src="assets/14b_examples/ref_2_motion1.gif" height="250"/></td>
    <td align="center" style="padding: 2px;"><img src="assets/14b_examples/ref_2_motion2.gif" height="250"/></td>
    <td align="center" style="padding: 2px;"><img src="assets/14b_examples/ref_2_motion3.gif" height="250"/></td>
  </tr>
</table>

<br>

#### 1.3B Model
The 1.3 B model also delivers strong performance (from 1.3b_2 ckpt).

<table align="center">
  <tr>
    <th style="text-align: center;">Reference</th>
    <th style="text-align: center;">Motion 1</th>
    <th style="text-align: center;">Motion 2</th>
    <th style="text-align: center;">Motion 3</th>
  </tr>
  <tr>
    <td align="center" style="padding: 2px;"><img src="./examples/new_examples/3.png" height="250"/></td>
    <td align="center" style="padding: 2px;"><img src="assets/1.3b_examples/ref3_motion1_1.3b.gif" height="250"/></td>
    <td align="center" style="padding: 2px;"><img src="assets/1.3b_examples/ref3_motion2_1.3b.gif" height="250"/></td>
    <td align="center" style="padding: 2px;"><img src="assets/1.3b_examples/ref3_motion3_1.3b.gif" height="250"/></td>
  </tr>
</table>

Also support longer video & out-of-domain cases 
<p align="center">
  <img src="./assets/1.3b_examples/combined_video1.gif" height="250"/> &nbsp;&nbsp;&nbsp;&nbsp; <img src="./assets/1.3b_examples/combined_video2.gif" height="250"/>
</p>

<br>



## 🔧 Dependencies and Installation

1. Clone Repo
    ```bash
    git clone https://github.com/ssj9596/One-to-All-Animation.git
    cd One-to-All-Animation
    ```

2. Create Conda Environment and Install Dependencies
    ```bash
    # create new conda env
    conda create -n one-to-all python=3.12
    conda activate one-to-all

    # install pytorch
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    # or
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -i https://mirrors.aliyun.com/pypi/simple/

    # install python dependencies
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/


    # (Recommended) install flash attention 3 (or 2) from source:
    # https://github.com/Dao-AILab/flash-attention
    ```

3. Download Models

   - Download pretrained models
   ```bash
    cd ./pretrained_models 
    bash download_pretrained_models.py
    ```

   - Download checkpoints
    ```bash
    cd ./checkpoints
    bash download_checkpoints.py
    ```

    > 💡 **Tip**: Edit the script and uncomment the specific models you want to download.
    > - **1.3B_1**: Best performance on video benchmark among 1.3B models (paper results).
    > - **1.3B_2**: Further trained on v1 with large camera movement data and increased image ratio. Better for dynamic video generation. Best on image benchmark (paper results). 
    > - **14B**: Best overall performance among 14B models (paper results).

<br>


## ☕️ Quick Inference

We provide several examples in the [`examples`](./examples) folder. 
Run the following commands to try it out:

```bash
# Step 1: Prepare model input
cd video-generation
python infer_preprocess.py

# Step 2: Run inference with your preferred model
python inference_1.3b.py  # For 1.3B model
# or
python inference_14b.py   # For 14B model
```
You can enter the script to modify the input path.

<br>

## 🎬 Training from scratch

>💡 **Data Collection Required**: We find current open-source datasets are not sufficient for training from scratch. We strongly recommend collecting *at least 3,000 additional high-quality video samples* for better results.

We divide the training process into several steps to help you train from scratch (using 1.3B as an example).

1. Download Pretrained Models

    Download the base model from HuggingFace: [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)

2. Download Training Datasets and Pose Pool

    ```bash
    cd datasets
    bash setup_datasets.sh
    ```
    
    This will download and prepare:
    - Training datasets (open-source + cartoon): `datasets/opensource_dataset/`
    - Pose pool for face enhancement: `datasets/opensource_pose_pool/`
    
    <details>
    <summary>Manual Download Links</summary>
    
    - [opensource_dataset](https://huggingface.co/datasets/MochunniaN1/One-to-All-sub/tree/main/opensource_dataset)
    - [opensource_pose_pool](https://huggingface.co/datasets/MochunniaN1/One-to-All-sub/tree/main/opensource_pose_pool)
    
    </details>

3. Training

    We provide three-stage training scripts:
    * Stage 1: Reference Extractor
    
    ```bash
    cd video-generation
    bash training_scripts/train1.3b_only_refextractor_2d.sh
    # Convert checkpoint to FP32
    cd outputs_wanx1.3b/train1.3b_only_refextractor_2d/checkpoint-xxx
    mkdir fp32_model_xxx
    python zero_to_fp32.py . fp32_model_xxx --safe_serialization
    # Run inference (update model path in inference_refextractor.py first)
    cd ../../../
    # Edit inference_refextractor.py and change ckpt_path to:
    # ./outputs_wanx1.3b/train1.3b_only_refextractor_2d/checkpoint-xxx/fp32_model_xxx
    python inference_refextractor.py
    ```
    
    * Stage 2: Pose Control
    ```bash 
    bash training_scripts/train1.3b_posecontrol_prefix_2d.sh
    ```
    * Stage 3: Token Replace for Long video generation
    ```bash 
    bash training_scripts/train1.3b_posecontrol_prefix_2d_tokenreplace.sh
    ```
    > 💡 **Training Notes**: 
    > - **Each stage uses different training resolutions** - check the scripts for specific resolution settings
    > - **Fine-tuning from our checkpoints**: If you want to continue training from our pre-trained models, directly use the *Stage 3 script* and modify the checkpoint path

<br>

## 📊 Reproduce Paper Results

We provide scripts to reproduce the quantitative results reported in our paper.

1. Download Benchmark
    ```bash
    cd benchmark
    bash setup_datasets.sh
    ```
2. Prepare Model Input
    ```bash
    cd ../video-generation
    python reproduce/infer_preprocess.py 
    ```
3. Run Inference

    We provide inference scripts for different model sizes and datasets:
    ```bash
    # TikTok dataset
    python reproduce/inference_tiktok1.3b.py   # 1.3B model
    python reproduce/inference_tiktok14b.py    # 14B model
    
    # Cartoon dataset
    python reproduce/inference_cartoon1.3b.py  # 1.3B model
    python reproduce/inference_cartoon14b.py   # 14B model

4. Prepare gt/pred pairs for Judge
   ```bash
   cd ../benchmark
   # TikTok dataset
   python prepare_eval_frames_tiktok.py
   # Cartoon dataset
   python prepare_eval_frames_cartoon.py
   ```

5. Run judge 
   ```bash
   # prepare DisCo environment and lpips fvd ckpt for judge
   cd DisCo
   # TikTok dataset
   bash eval_tiktok.sh
   python summary.py
   ```

<br>

## Acknowledgments

Our project is based on [opensora](https://github.com/hpcaitech/Open-Sora). Some codes are brought from [StableAnimator](https://github.com/Francis-Rings/StableAnimator) and [Wan-Animate](https://github.com/Wan-Video/Wan2.2). Thanks for their awesome works.


## 📧 Contact
If you have any questions, please feel free to reach us at `ssj180123@gmail.com`

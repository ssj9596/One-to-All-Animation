import os
# Set HuggingFace mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

os.makedirs("Wan2.1-T2V-1.3B-Diffusers", exist_ok=True)
os.makedirs("Wan2.1-T2V-14B-Diffusers", exist_ok=True)

print("Using HF Mirror: https://hf-mirror.com")
print("="*50)

# dwpose
snapshot_download(
    repo_id="FrancisRing/StableAnimator",
    allow_patterns="DWPose/*",
    local_dir="./",
    local_dir_use_symlinks=False,
    repo_type="model",
    resume_download=True,
)

# wanpose
snapshot_download(
    repo_id="Wan-AI/Wan2.2-Animate-14B",
    allow_patterns="process_checkpoint/det/*",
    local_dir="./",
    local_dir_use_symlinks=False,
    repo_type="model",
    resume_download=True,
)
snapshot_download(
    repo_id="Wan-AI/Wan2.2-Animate-14B",
    allow_patterns="process_checkpoint/pose2d/*",
    local_dir="./",
    local_dir_use_symlinks=False,
    repo_type="model",
    resume_download=True,
)

# # 1.3b 
# snapshot_download(
#     repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
#     local_dir="./Wan2.1-T2V-1.3B-Diffusers",
#     local_dir_use_symlinks=False,
#     repo_type="model",
#     resume_download=True,
# )

# # 14b 
# snapshot_download(
#     repo_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
#     local_dir="./Wan2.1-T2V-14B-Diffusers",
#     local_dir_use_symlinks=False,
#     repo_type="model",
#     resume_download=True,
# )



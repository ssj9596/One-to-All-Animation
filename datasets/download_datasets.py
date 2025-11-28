import os
# Set HuggingFace mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

os.makedirs("opensource_dataset", exist_ok=True)
os.makedirs("opensource_pose_pool", exist_ok=True)

print("Using HF Mirror: https://hf-mirror.com")
print("="*50)

try:
    # 1. Download training datasets
    print("\n[1/2] Downloading training datasets...")
    snapshot_download(
        repo_id="MochunniaN1/One-to-All-sub",
        allow_patterns="opensource_dataset/*",
        local_dir="./",
        local_dir_use_symlinks=False,
        repo_type="dataset"
    )
    print("✓ Training datasets downloaded successfully!")
    
    # 2. Download pose pool
    print("\n[2/2] Downloading pose pool...")
    snapshot_download(
        repo_id="MochunniaN1/One-to-All-sub",
        allow_patterns="opensource_pose_pool/*",
        local_dir="./",
        local_dir_use_symlinks=False,
        repo_type="dataset"
    )
    print("✓ Pose pool downloaded successfully!")
    
    print("\n" + "="*50)
    print("All downloads completed successfully!")
    print("="*50)
    
except Exception as e:
    print(f"\n❌ Error occurred: {str(e)}")
    print("Please check your internet connection.")

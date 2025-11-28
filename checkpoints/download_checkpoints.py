import os
# Set HuggingFace mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# export HF_ENDPOINT=https://hf-mirror.com
from huggingface_hub import snapshot_download

os.makedirs("One-to-All-14b", exist_ok=True)
os.makedirs("One-to-All-1.3b_1", exist_ok=True)
os.makedirs("One-to-All-1.3b_2", exist_ok=True)

print("Using HF Mirror: https://hf-mirror.com")
print("="*50)

 
# 1.3b 
# snapshot_download(
#     repo_id="MochunniaN1/One-to-All-1.3b_1",
#     local_dir="./One-to-All-1.3b_1",
#     local_dir_use_symlinks=False,
#     repo_type="model",
#     resume_download=True,
    
# )

snapshot_download(
    repo_id="MochunniaN1/One-to-All-1.3b_2",
    local_dir="./One-to-All-1.3b_2",
    local_dir_use_symlinks=False,
    repo_type="model",
    resume_download=True,
)

# 14b 
# snapshot_download(
#     repo_id="MochunniaN1/One-to-All-14b",
#     local_dir="./One-to-All-14b",
#     local_dir_use_symlinks=False,
#     repo_type="model",
#     resume_download=True,
# )


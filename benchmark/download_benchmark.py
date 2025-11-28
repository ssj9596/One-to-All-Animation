import os
import zipfile
from pathlib import Path

# Set HuggingFace mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

print("Using HF Mirror: https://hf-mirror.com")
print("="*50)

def extract_zip(zip_path, extract_to="."):
    """Extract zip file and remove it after extraction"""
    print(f"\n📦 Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")
        
        # Remove zip file after extraction
        # os.remove(zip_path)
        # print(f"✓ Removed {zip_path}")
    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")

# Download datasets
datasets = [
    "tiktok_test_set.zip",
    "cartoon_test_set.zip"
]

for dataset in datasets:
    print(f"\n{'='*50}")
    print(f"📥 Downloading {dataset}...")
    print(f"{'='*50}")
    
    snapshot_download(
        repo_id="MochunniaN1/One-to-All-sub",
        local_dir=".",
        allow_patterns=dataset,
        local_dir_use_symlinks=False,
        repo_type="dataset",
        resume_download=True,
    )

    zip_path = Path(dataset)
    if zip_path.exists():
        extract_zip(zip_path)
    else:
        print(f"⚠️ {dataset} not found, skipping extraction")

print("\n" + "="*50)
print("✓ All datasets downloaded and extracted!")
print("="*50)

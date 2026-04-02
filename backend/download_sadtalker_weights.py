import os
import requests
from tqdm import tqdm

def download_file(url, target_path):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"   ✅ Already exists: {os.path.basename(target_path)}")
        return
    
    print(f"   ⬇️  Downloading: {os.path.basename(target_path)} from {url}...")
    try:
        response = requests.get(url, stream=True, allow_redirects=True, timeout=60)
        response.raise_for_status() # Check for HTTP errors
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as file, tqdm(
            desc=os.path.basename(target_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        print(f"   ❌ Error downloading {os.path.basename(target_path)}: {e}")
        if os.path.exists(target_path): os.remove(target_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "SadTalker", "checkpoints")
GFPGAN_DIR = os.path.join(BASE_DIR, "SadTalker", "gfpgan", "weights")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GFPGAN_DIR, exist_ok=True)

# Using official Winfredy repo URLs which are more stable for these binary releases
# Note: SadTalker_V0.0.2_256.safetensors is sometimes renamed or was in OpenTalker
models = {
    # SadTalker Main Checkpoints
    "SadTalker_V0.0.2_256.safetensors": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors",
    "SadTalker_V0.0.2_512.safetensors": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors",
    "mapping_00109-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
    "mapping_00229-model.pth.tar": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
    
    # BFM Fitting (Crucial!)
    "BFM_Fitting.zip": "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip",
}

enhancer_models = {
    "alignment_WFLW_4HG.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
    "detection_Resnet50_Final.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "parsing_parsenet.pth": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
}

print("📦 Downloading SadTalker Checkpoints...")
for name, url in models.items():
    download_file(url, os.path.join(CHECKPOINT_DIR, name))

print("\n✨ Downloading Enhancer Weights...")
for name, url in enhancer_models.items():
    download_file(url, os.path.join(GFPGAN_DIR, name))

# Extract BFM Fitting
bfm_zip = os.path.join(CHECKPOINT_DIR, "BFM_Fitting.zip")
if os.path.exists(bfm_zip) and not os.path.exists(os.path.join(CHECKPOINT_DIR, "BFM_Fitting")):
    import zipfile
    print("   📂 Extracting BFM_Fitting.zip...")
    try:
        with zipfile.ZipFile(bfm_zip, 'r') as zip_ref:
            zip_ref.extractall(CHECKPOINT_DIR)
        os.remove(bfm_zip)
    except Exception as e:
        print(f"   ❌ Failed to extract BFM_Fitting.zip: {e}")

print("\n✅ All SadTalker weights Ready!")

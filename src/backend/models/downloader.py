import os
import requests
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Base URL for the GitHub Release containing the models
# User should upload models to a release tagged v0.1.0
GITHUB_RELEASE_URL = "https://github.com/OleksiiKrasnoshtanov240247/sign-language/releases/download/v0.1.0/"

MODELS = {
    "static/best_model.pth": "static_best_model.pth",
    "static/label_encoder.pkl": "static_label_encoder.pkl",
    "dynamic/best_model.pth": "dynamic_best_model.pth",
    "dynamic/classes.npy": "dynamic_classes.npy",
    "mediapipe/hand_landmarker.task": "hand_landmarker.task"
}

def download_file(url: str, dest_path: Path):
    """Download a file with a progress bar."""
    print(f"Downloading {dest_path.name}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if tqdm:
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            
        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                if tqdm:
                    progress_bar.update(len(data))
                file.write(data)
                
        if tqdm:
            progress_bar.close()
            if total_size != 0 and progress_bar.n != total_size:
                print("ERROR, something went wrong with the download")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        # Remove the partial file if it exists
        if dest_path.exists():
            dest_path.unlink()

def check_and_download_models():
    """Check if model files exist, download them if they don't."""
    project_root = Path(__file__).parent.parent.parent.parent
    models_dir = project_root / "models"
    
    missing_models = []
    for local_path, remote_filename in MODELS.items():
        full_local_path = models_dir / local_path
        if not full_local_path.exists():
            missing_models.append((full_local_path, remote_filename))
            
    if not missing_models:
        return # All models present
        
    print(f"Missing {len(missing_models)} model files. Downloading from GitHub Releases...")
    for local_path, remote_filename in missing_models:
        url = GITHUB_RELEASE_URL + remote_filename
        download_file(url, local_path)
    
    print("Model download complete.")

if __name__ == "__main__":
    check_and_download_models()

import os
import requests
from . import config

def download_assets():
    """
    Downloads all necessary assets: model weights and language packages.
    """
    os.makedirs(config.ASSETS_DIR, exist_ok=True)
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)

    if not os.path.exists(config.IMAGE_PATH):
        try:
            response = requests.get(config.IMAGE_URL)
            response.raise_for_status()
            with open(config.IMAGE_PATH, "wb") as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error downloading image: {e}")

    if not os.path.exists(config.ENHANCER_WEIGHTS_PATH):
        try:
            response = requests.get(config.ENHANCER_WEIGHTS_URL, stream=True)
            response.raise_for_status()
            with open(config.ENHANCER_WEIGHTS_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error downloading weights: {e}")
    
    # Translation models (IndicTrans2, NLLB, MarianMT) are downloaded automatically
    # by HuggingFace transformers when first used
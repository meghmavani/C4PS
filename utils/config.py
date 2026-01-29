import os
from dotenv import load_dotenv
load_dotenv()

# --- DIRECTORY CONFIGS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
ARGOS_PACKAGES_DIR = os.getenv("ARGOS_PACKAGES_DIR")


# --- DOWNLOADER CONFIGS ---
IMAGE_URL = os.getenv("IMAGE_URL")
ENHANCER_WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
IMAGE_PATH = os.path.join(ASSETS_DIR, "image.jpg")
ENHANCER_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'RealESRGAN_x4plus.pth')
CAPTIONING_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'transformer_model_cache')

# --- TRANSLATION CONFIGS ---
TARGET_LANGUAGES = [
    "sq", "ar", "az", "eu", "bg", "ca", "zh", "zt", "cs", "da",
    "nl", "eo", "et", "fi", "fr", "gl", "de", "el", "he", "hu",
    "id", "ga", "it", "ja", "ko", "ky", "lv", "lt", "ms", "nb",
    "fa", "pl", "pt", "pb", "ro", "ru", "sk", "sl", "es", "sv",
    "tl", "th", "tr", "uk", "vi"
]

# --- ONLINE OUTPUT CONFIGS ---
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DISCORD_SERVER_INVITE = os.getenv("DISCORD_SERVER_INVITE")
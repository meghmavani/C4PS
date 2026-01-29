import os
import requests
from . import config

# ============================================================
# FORCE ARGOS PACKAGE DIRECTORY (CRITICAL)
# ============================================================

packages_dir = config.ARGOS_PACKAGES_DIR
if not packages_dir:
    raise RuntimeError(
        "ARGOS_PACKAGES_DIR is not set. "
        "Please define it in your .env file."
    )

# Make absolute if relative
if not os.path.isabs(packages_dir):
    packages_dir = os.path.join(config.BASE_DIR, packages_dir)

os.environ["ARGOS_PACKAGES_DIR"] = packages_dir
os.makedirs(packages_dir, exist_ok=True)

print("[INFO] Using ARGOS_PACKAGES_DIR =", packages_dir)

# ============================================================
# Safe to import Argos AFTER env var is set
# ============================================================

import argostranslate.package
import argostranslate.translate


def download_assets():
    """
    Downloads all required assets:
    - Sample image
    - Real-ESRGAN weights
    - Argos Translate language models (offline translation)
    """

    # --------------------------------------------------------
    # Ensure base directories exist
    # --------------------------------------------------------
    os.makedirs(config.ASSETS_DIR, exist_ok=True)
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Download sample image (if missing)
    # --------------------------------------------------------
    if not os.path.exists(config.IMAGE_PATH):
        try:
            print("[INFO] Downloading sample image...")
            response = requests.get(config.IMAGE_URL, timeout=30)
            response.raise_for_status()
            with open(config.IMAGE_PATH, "wb") as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to download image: {e}")

    # --------------------------------------------------------
    # Download Real-ESRGAN weights (if missing)
    # --------------------------------------------------------
    if not os.path.exists(config.ENHANCER_WEIGHTS_PATH):
        try:
            print("[INFO] Downloading Real-ESRGAN weights...")
            response = requests.get(
                config.ENHANCER_WEIGHTS_URL,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            with open(config.ENHANCER_WEIGHTS_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to download enhancer weights: {e}")

    # --------------------------------------------------------
    # Install Argos Translate language models
    # --------------------------------------------------------
    print("[INFO] Checking Argos Translate language models...")

    try:
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()

        source_lang = "en"

        for target_lang in config.TARGET_LANGUAGES:
            # Skip if already installed
            try:
                if argostranslate.translate.get_translation_from_codes(
                    source_lang, target_lang
                ):
                    print(f"[OK] Argos model already installed: en → {target_lang}")
                    continue
            except Exception:
                pass

            # Find matching package
            pkg = next(
                (
                    p for p in available_packages
                    if p.from_code == source_lang and p.to_code == target_lang
                ),
                None
            )

            if not pkg:
                print(f"[WARN] No Argos package found for en → {target_lang}")
                continue

            print(f"[INSTALL] Installing Argos model: en → {target_lang}")
            argostranslate.package.install_from_path(pkg.download())

        print("[INFO] Argos Translate setup complete.")

    except Exception as e:
        print(f"[ERROR] Argos Translate installation failed: {e}")


if __name__ == "__main__":
    download_assets()

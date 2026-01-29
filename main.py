import sys
from torchvision.transforms.v2 import functional as F

# --- MONKEY-PATCH FIX ---
sys.modules['torchvision.transforms.functional_tensor'] = F
# --- END OF FIX ---

from utils.terminal_ui import print_header, print_step, suppress_warnings, clear_screen
suppress_warnings()

import os
import torch
from PIL import Image
import inquirer

from utils.downloader import download_assets
from utils import config
from enhancement.enhancer import enhance_image
from captioning.generator import CaptionGenerator
from translation.translator import (
    translate_caption,
    INDIC_LANGUAGES,
    INTERNATIONAL_LANGUAGES
)
from utils.output_handler import send_to_discord, display_offline_report

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ============================================================
# LANGUAGE SELECTION (CLEAN & MINIMAL)
# ============================================================
def get_language_category_from_terminal():
    questions = [
        inquirer.List(
            'category',
            message="[?] Choose language category",
            choices=[
                "Indic Languages",
                "International Languages"
            ]
        )
    ]
    answers = inquirer.prompt(questions)
    return "indic" if "Indic" in answers["category"] else "international"

def get_language_choice_from_terminal(category):
    languages = INDIC_LANGUAGES if category == "indic" else INTERNATIONAL_LANGUAGES

    choices = [
        f"{idx}. {info['name']}"
        for idx, info in languages.items()
    ]

    questions = [
        inquirer.List(
            'lang_choice',
            message="[?] Choose target language",
            choices=choices
        )
    ]

    answers = inquirer.prompt(questions)
    return int(answers["lang_choice"].split(".")[0])

# ============================================================
# PIPELINE
# ============================================================
def run_pipeline():
    clear_screen()
    print_header()

    print_step(0, "Setting up assets...")
    download_assets()

    device = get_device()
    caption_model = CaptionGenerator(device)

    print_step(1, "Enhancing image...")
    enhanced_image = enhance_image(
        config.IMAGE_PATH,
        mode="general",
        tile_size=800,
        face_enhance=False
    )

    print_step(2, "Generating image caption...")
    english_caption = caption_model.generate_caption(enhanced_image)

    print_step(3, "Saving enhanced image...")
    base, ext = os.path.splitext(config.IMAGE_PATH)
    enhanced_path = f"{base}_enhanced{ext}"
    enhanced_image.save(enhanced_path)

    print_step(4, "Choosing translation language...")
    category = get_language_category_from_terminal()
    choice = get_language_choice_from_terminal(category)

    print_step(5, "Translating caption...")
    multilingual_captions = translate_caption(
        english_caption,
        category,
        choice
    )

    print_step(6, "Finalizing output...")
    display_offline_report(
        enhanced_path,
        english_caption,
        multilingual_captions
    )

if __name__ == "__main__":
    run_pipeline()

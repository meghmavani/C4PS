import sys
from utils import config
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
from tqdm import tqdm
import requests
import tempfile
import shutil

from utils.downloader import download_assets
from enhancement.enhancer import enhance_image
from captioning.generator import CaptionGenerator
from translation.translator import translate_caption
from utils.output_handler import send_to_discord, display_offline_report, is_discord_upload_too_large


# --- THIS FUNCTION IS UPDATED ---
def get_image_source_from_terminal():
    """Asks the user for the image source (local path or URL) using standard input."""
    while True:
        try:
            # Use Python's built-in input()
            source = input("[?] Enter the path to a local image OR a URL: ").strip()
            if source: # Check if the input is not empty after stripping whitespace
                return source
            else:
                print("[ERROR] Input cannot be empty. Please try again.")
        except EOFError: # Handle Ctrl+D or unexpected end of input
             print("\nOperation cancelled by user. Exiting.")
             exit()
        except KeyboardInterrupt: # Handle Ctrl+C
            print("\nOperation cancelled by user. Exiting.")
            exit()
# --- END UPDATED FUNCTION ---


def download_image_if_url(source):
    """Checks if the source is a URL, downloads it, and returns a local path."""
    if source.startswith(('http://', 'https://')):
        print(f"[INFO] Downloading image from URL: {source}")
        try:
            response = requests.get(source, stream=True, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes

            # Create a temporary file to save the download
            # Use a known extension like .jpg for compatibility
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            with open(temp_file.name, 'wb') as f:
                 shutil.copyfileobj(response.raw, f)
            print(f"[INFO] Image downloaded temporarily to: {temp_file.name}")
            return temp_file.name, True # Return path and flag indicating it was downloaded
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to download image from URL: {e}")
            return None, False
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during download: {e}")
            return None, False
    elif os.path.exists(source):
        print(f"[INFO] Using local image path: {source}")
        return source, False # It's a local path, not downloaded
    else:
        print(f"[ERROR] Input is not a valid URL or existing local file path: {source}")
        return None, False


# ...(get_device, get_output_mode_from_terminal, etc. remain the same)...
def get_device():
    """Checks for the best available hardware and returns the device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_output_mode_from_terminal():
    """Asks the user to choose an output mode using a terminal menu."""
    questions = [
        inquirer.List('mode',
                      message="[?] Choose output mode",
                      choices=['Online (Post to Discord)', 'Offline (Generate local report)'],
                      ),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        print("No selection made. Exiting.")
        exit()
    return 'online' if 'Online' in answers['mode'] else 'offline'

def get_enhancement_mode_from_terminal():
    """Asks the user to choose an enhancement mode."""
    questions = [
        inquirer.List('mode',
                      message="[?] Choose enhancement quality",
                      choices=[
                          # Added [-> 4x], [-> 2x (or 8x)] tags
                          'Vehicle Optimized (Auto-Detect) | (Slower) | Best for mixed-content posts [-> 4x]',
                          'Sharp (Vehicles/Anime)          | ~10 sec  | Manually force sharp mode     [-> 4x]',
                          'General (x4plus)                | ~150 sec | Manually force general/people [-> 4x]',
                          'Fast (x2)                       | ~2 sec   | Fastest, good quality         [-> 2x (or 8x if chained)]'
                      ],
                      ),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        print("No selection made. Exiting.")
        exit()

    # Map the user-friendly string to a simple key
    if 'Fast (x2)' in answers['mode']:
        return 'fast'
    if 'Sharp (Vehicles/Anime)' in answers['mode']:
        return 'sharp_anime'
    if 'Vehicle Optimized (Auto-Detect)' in answers['mode']:
        return 'auto_vehicle'

    # Default to the general model
    return 'general'

def get_tile_size_from_terminal():
    """Asks the user to choose a tile size."""
    questions = [
        inquirer.List('tile_size',
                      message="[?] Choose tile size (Smaller uses less RAM; 800 is balanced)",
                      choices=[
                          'Medium (800)',
                          'Small (400)',
                          'Large (1200)'
                      ],
                      default='Medium (800)',
                      ),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        return 800  # Default if they exit

    if 'Small' in answers['tile_size']:
        return 400
    if 'Large' in answers['tile_size']:
        return 1200
    return 800  # Default

def get_face_enhance_prompt():
    """Asks the user if they want to enhance faces (GFPGAN) using a list."""
    questions = [
        inquirer.List('face_enhance',
                      message="[?] This model is for general photos. Enhance faces too? (Slower)",
                      choices=[
                          'No (Default)',
                          'Yes (Enhance Faces)'
                      ],
                      default='No (Default)',
                      ),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        return False
    return 'Yes' in answers['face_enhance']

def get_chain_enhance_prompt():
    """Asks the user if they want to upscale the 2x output again."""
    questions = [
        inquirer.List('chain_enhance',
                      message="[?] You chose 2x. Upscale AGAIN with 4x ( slower, may reduce quality)?",
                      choices=[
                          'No (Keep 2x)',
                          'Yes (Upscale again to 8x)' # Clarified the output size
                      ],
                      default='No (Keep 2x)',
                      ),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        return False
    return 'Yes' in answers['chain_enhance']


def get_discord_oversize_action_from_terminal():
    """Asks how to proceed when an image is too large for Discord webhook upload."""
    questions = [
        inquirer.List(
            'oversize_action',
            message="[?] Enhanced image may exceed Discord upload limits. What should we do?",
            choices=[
                'Show local report instead',
                'Send compressed version to Discord',
                'Cancel this run'
            ],
            default='Send compressed version to Discord'
        ),
    ]
    answers = inquirer.prompt(questions)
    if not answers:
        return 'cancel'

    selected = answers['oversize_action'].lower()
    if 'compressed' in selected:
        return 'compressed'
    if 'local' in selected:
        return 'local'
    return 'cancel'

def get_target_languages_from_terminal():
    """Asks the user to choose target languages for translation."""
    LANG_OPTIONS = [
        # Indian Languages (NLLB)
        ("Hindi", "hi"),
        ("Tamil", "ta"),
        ("Telugu", "te"),
        ("Kannada", "kn"),
        ("Malayalam", "ml"),
        ("Bengali", "bn"),
        ("Marathi", "mr"),
        ("Gujarati", "gu"),
        ("Punjabi", "pa"),
        ("Odia", "or"),
        ("Assamese", "as"),
        ("Urdu", "ur"),
        
        # Major World Languages (MarianMT)
        ("French", "fr"),
        ("Spanish", "es"),
        ("German", "de"),
        ("Chinese (Simplified)", "zh"),
        ("Japanese", "ja"),
        ("Russian", "ru"),
        ("Arabic", "ar"),
        ("Portuguese", "pt"),
        ("Italian", "it"),
        ("Dutch", "nl"),
        ("Korean", "ko"),
        ("Turkish", "tr"),
        ("Vietnamese", "vi"),
        ("Indonesian", "id")
    ]
    
    choices = [f"{name} ({code.upper()})" for name, code in LANG_OPTIONS]
    
    questions = [
        inquirer.Checkbox('languages',
            message="[?] Select target languages (Space to select, Enter to confirm)",
            choices=choices,
            default=["Hindi (HI)", "French (FR)"]
        ),
    ]
    
    # Optional: Allow skipping
    print("\n[INFO] Press Enter without selecting to default to English only.")
    answers = inquirer.prompt(questions)
    
    if not answers or not answers['languages']:
        print("[INFO] No languages selected. Using English only.")
        return []

    selected_codes = []
    # Map back to codes
    for selection in answers['languages']:
        # Extract code from "Name (CODE)" using rsplit to find the LAST parenthesis
        # "Chinese (Simplified) (ZH)" -> "ZH"
        code = selection.rsplit('(', 1)[1].strip(')').lower()
        selected_codes.append(code)
        
    return selected_codes


def run_pipeline():
    clear_screen()

    print_header()
    # Get image source using the new function
    image_source = get_image_source_from_terminal()
    image_path, is_downloaded = download_image_if_url(image_source)
    if image_path is None:
        return # Exit if download or validation failed

    # Continue with the rest of the prompts
    output_mode = get_output_mode_from_terminal()
    enhancement_mode = get_enhancement_mode_from_terminal()
    tile_size = get_tile_size_from_terminal()

    enhance_faces = False
    if enhancement_mode == 'general':
        enhance_faces = get_face_enhance_prompt()

    chain_enhance = False
    if enhancement_mode == 'fast':
        chain_enhance = get_chain_enhance_prompt()

    # Get Languages
    target_languages_list = get_target_languages_from_terminal()

    print_step(0, "Setting up assets...")
    download_assets()

    device = get_device()
    caption_model = CaptionGenerator(device)
    english_caption = ""
    enhanced_image = None
    image_to_caption = None

    # --- Pipeline Logic using image_path ---
    if enhancement_mode == 'fast':
        # --- PATH 1: Fast (x2), maybe chained ---
        print_step(1, f"Enhancing image (Mode: {enhancement_mode})...")
        with tqdm(total=1, desc="Enhancing (2x pass)") as pbar:
            enhanced_image_2x = enhance_image(
                image_path, mode='fast',
                tile_size=tile_size, enhance_faces=False
            )
            pbar.update(1)

        if chain_enhance:
            print_step("1b", "Upscaling 2x output again with 4x model...")
            temp_2x_path = None
            try:
                temp_2x_path = "temp_2x_output.png"
                enhanced_image_2x.save(temp_2x_path)
                with tqdm(total=1, desc="Enhancing (4x pass)") as pbar:
                    enhanced_image = enhance_image(
                        temp_2x_path, mode='general',
                        tile_size=tile_size, enhance_faces=enhance_faces
                    )
                    pbar.update(1)
            finally:
                if temp_2x_path and os.path.exists(temp_2x_path):
                    try:
                        os.remove(temp_2x_path)
                    except OSError:
                         print(f"[WARNING] Could not delete temporary file: {temp_2x_path}")

            image_to_caption = enhanced_image
        else:
            enhanced_image = enhanced_image_2x
            image_to_caption = enhanced_image

        print_step(2, "Generating image caption (from enhanced image)...")
        english_caption = caption_model.generate_caption(image_to_caption)

    elif enhancement_mode == 'auto_vehicle':
         # --- PATH 2: Auto-Vehicle ---
        print_step(1, f"Enhancing image (Mode: {enhancement_mode})...")
        with tqdm(total=1, desc="Enhancing (Auto-Vehicle)") as pbar:
            enhanced_image = enhance_image(
                image_path, mode='auto_vehicle',
                tile_size=tile_size, enhance_faces=False
            )
            pbar.update(1)
        print_step(2, "Generating image caption (from enhanced image)...")
        english_caption = caption_model.generate_caption(enhanced_image)

    else:
        # --- PATH 3: (General x4plus / Sharp-Anime) ---
        print_step(1, "Generating image caption (from original)...")
        try:
            original_image_pil = Image.open(image_path)
            image_to_caption = original_image_pil
            english_caption = caption_model.generate_caption(image_to_caption)
        except Exception as e:
            print(f"[ERROR] Could not open image for captioning at {image_path}: {e}")
            if is_downloaded and os.path.exists(image_path):
                 try: os.remove(image_path)
                 except OSError: pass
            return

        print(f"[INFO] Generated caption: \"{english_caption}\"")

        print_step(2, f"Enhancing image (Mode: {enhancement_mode})...")
        with tqdm(total=1, desc=f"Enhancing ({enhancement_mode})") as pbar:
            enhanced_image = enhance_image(
                image_path, mode=enhancement_mode,
                tile_size=tile_size, enhance_faces=enhance_faces
            )
            pbar.update(1)

    # --- PIPELINE REJOINS ---
    try:
        if is_downloaded:
             source_basename = os.path.basename(requests.utils.urlparse(image_source).path)
             if not source_basename: source_basename = "downloaded_image"
             base, ext = os.path.splitext(source_basename)
        else:
             base, ext = os.path.splitext(os.path.basename(image_path))
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        enhanced_image_path = os.path.join(output_dir, f"{base}_enhanced{ext if ext else '.jpg'}")

    except Exception:
        enhanced_image_path = "assets/enhanced_output.jpg"

    enhanced_image.save(enhanced_image_path)
    print(f"[INFO] Enhanced image saved to: {enhanced_image_path}")


    print_step(3, "Translating captions...")
    multilingual_captions = translate_caption(
        english_caption,
        target_languages=target_languages_list
    )

    print_step(4, f"Finalizing output (Mode: {output_mode})...")
    if output_mode == 'online':
        if is_discord_upload_too_large(enhanced_image_path):
            size_mb = os.path.getsize(enhanced_image_path) / (1024 * 1024)
            print(f"[WARNING] Image size is {size_mb:.2f} MB; Discord webhooks often reject large uploads.")
            oversize_action = get_discord_oversize_action_from_terminal()

            if oversize_action == 'local':
                display_offline_report(
                    enhanced_image_path,
                    english_caption,
                    multilingual_captions
                )
            elif oversize_action == 'cancel':
                print("[INFO] Upload cancelled by user.")
                if is_downloaded and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        print(f"[INFO] Deleted temporary file: {image_path}")
                    except OSError as e:
                        print(f"[WARNING] Could not delete temporary file {image_path}: {e}")
                return
            else:
                send_to_discord(
                    config.DISCORD_WEBHOOK_URL,
                    config.DISCORD_SERVER_INVITE,
                    enhanced_image_path,
                    english_caption,
                    multilingual_captions,
                    allow_compression=True
                )
        else:
            send_to_discord(
                config.DISCORD_WEBHOOK_URL,
                config.DISCORD_SERVER_INVITE,
                enhanced_image_path,
                english_caption,
                multilingual_captions,
                allow_compression=False
            )
    else:
        display_offline_report(
            enhanced_image_path,
            english_caption,
            multilingual_captions
        )

    # Clean up temporary downloaded file
    if is_downloaded and os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"[INFO] Deleted temporary file: {image_path}")
        except OSError as e:
            print(f"[WARNING] Could not delete temporary file {image_path}: {e}")


if __name__ == '__main__':
    run_pipeline()
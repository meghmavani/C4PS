import os
import tempfile
from datetime import datetime
from discord import SyncWebhook, File, Embed
from PIL import Image
from .terminal_ui import display_online_report, display_offline_report


# Webhook uploads are commonly limited to ~8 MB on standard Discord setups.
DISCORD_MAX_UPLOAD_BYTES = 7_900_000


def is_discord_upload_too_large(image_path):
    """Return True if the file exceeds the conservative Discord webhook limit."""
    try:
        return os.path.getsize(image_path) > DISCORD_MAX_UPLOAD_BYTES
    except OSError:
        return False


def _prepare_image_for_discord(image_path):
    """Return an image path that is likely to fit Discord upload limits."""
    try:
        if os.path.getsize(image_path) <= DISCORD_MAX_UPLOAD_BYTES:
            return image_path, None
    except OSError:
        return image_path, None

    try:
        img = Image.open(image_path).convert("RGB")
        base_w, base_h = img.size

        # Progressive resize + quality reduction until under limit.
        scales = [1.0, 0.85, 0.7, 0.6, 0.5, 0.4]
        qualities = [90, 80, 70, 60, 50, 40]

        for scale in scales:
            new_w = max(256, int(base_w * scale))
            new_h = max(256, int(base_h * scale))
            work_img = img if scale == 1.0 else img.resize((new_w, new_h), Image.LANCZOS)

            for quality in qualities:
                fd, temp_path = tempfile.mkstemp(prefix="c4ps_discord_", suffix=".jpg")
                os.close(fd)
                work_img.save(temp_path, format="JPEG", quality=quality, optimize=True)
                if os.path.getsize(temp_path) <= DISCORD_MAX_UPLOAD_BYTES:
                    return temp_path, temp_path
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
    except Exception as e:
        print(f"[WARNING] Could not optimize image for Discord upload: {e}")

    return image_path, None

def send_to_discord(webhook_url, server_invite, enhanced_path, english_caption, multilingual_captions, allow_compression=True):
    """Sends the results to Discord and displays a report in the terminal."""
    if not webhook_url:
        print("[ERROR] Discord Webhook URL not found in .env file. Skipping online post.")
        return

    temp_upload_path = None
    try:
        webhook = SyncWebhook.from_url(webhook_url)
        embed = Embed(
            title="Image Captioning Result",
            description=f"**English Caption:**\n> {english_caption}",
            color=5814783
        )
        translations_text = ""
        for lang, text in multilingual_captions.items():
            translations_text += f"**{lang.upper()}:** {text}\n"
        embed.add_field(name="Multilingual Captions", value=translations_text)
        embed.set_footer(text=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if allow_compression:
            upload_path, temp_upload_path = _prepare_image_for_discord(enhanced_path)
        else:
            upload_path = enhanced_path
        image_file = File(upload_path, filename=os.path.basename(upload_path))
        
        message = webhook.send(
            file=image_file,
            username="C4PS Pipeline",
            embed=embed,
            wait=True
        )
        
        display_online_report(message.jump_url, server_invite)

    except Exception as e:
        print(f"[ERROR] Failed to send results to Discord: {e}")
    finally:
        if temp_upload_path and os.path.exists(temp_upload_path):
            try:
                os.remove(temp_upload_path)
            except OSError:
                pass


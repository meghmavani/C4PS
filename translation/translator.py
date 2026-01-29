import os
import torch
import argostranslate.translate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Suppress Argos warning
os.environ["ARGOS_DEVICE_TYPE"] = "cuda"

# ============================================================
# INDIC LANGUAGES (IndicBARTSS) — Alphabetical
# ============================================================
INDIC_LANGUAGES = {
    1: {"name": "Assamese", "code": "as"},
    2: {"name": "Bengali", "code": "bn"},
    3: {"name": "Gujarati", "code": "gu"},
    4: {"name": "Hindi", "code": "hi"},
    5: {"name": "Kannada", "code": "kn"},
    6: {"name": "Malayalam", "code": "ml"},
    7: {"name": "Marathi", "code": "mr"},
    8: {"name": "Odia", "code": "or"},
    9: {"name": "Punjabi", "code": "pa"},
    10: {"name": "Tamil", "code": "ta"},
    11: {"name": "Telugu", "code": "te"},
    12: {"name": "Urdu", "code": "ur"},
}

# ==============================================w==============
# INTERNATIONAL LANGUAGES (Argos)
# ============================================================
INTERNATIONAL_LANGUAGES = {
    1: {"name": "Albanian", "code": "sq"},
    2: {"name": "Arabic", "code": "ar"},
    3: {"name": "Azerbaijani", "code": "az"},
    4: {"name": "Basque", "code": "eu"},
    5: {"name": "Bulgarian", "code": "bg"},
    6: {"name": "Catalan", "code": "ca"},
    7: {"name": "Chinese", "code": "zh"},
    8: {"name": "Chinese (Traditional)", "code": "zt"},
    9: {"name": "Czech", "code": "cs"},
    10: {"name": "Danish", "code": "da"},
    11: {"name": "Dutch", "code": "nl"},
    12: {"name": "Esperanto", "code": "eo"},
    13: {"name": "Estonian", "code": "et"},
    14: {"name": "Finnish", "code": "fi"},
    15: {"name": "French", "code": "fr"},
    16: {"name": "Galician", "code": "gl"},
    17: {"name": "German", "code": "de"},
    18: {"name": "Greek", "code": "el"},
    19: {"name": "Hebrew", "code": "he"},
    20: {"name": "Hungarian", "code": "hu"},
    21: {"name": "Indonesian", "code": "id"},
    22: {"name": "Irish", "code": "ga"},
    23: {"name": "Italian", "code": "it"},
    24: {"name": "Japanese", "code": "ja"},
    25: {"name": "Korean", "code": "ko"},
    26: {"name": "Kyrgyz", "code": "ky"},
    27: {"name": "Latvian", "code": "lv"},
    28: {"name": "Lithuanian", "code": "lt"},
    29: {"name": "Malay", "code": "ms"},
    30: {"name": "Norwegian", "code": "nb"},
    31: {"name": "Persian", "code": "fa"},
    32: {"name": "Polish", "code": "pl"},
    33: {"name": "Portuguese", "code": "pt"},
    34: {"name": "Portuguese (Brazil)", "code": "pb"},
    35: {"name": "Romanian", "code": "ro"},
    36: {"name": "Russian", "code": "ru"},
    37: {"name": "Slovak", "code": "sk"},
    38: {"name": "Slovenian", "code": "sl"},
    39: {"name": "Spanish", "code": "es"},
    40: {"name": "Swedish", "code": "sv"},
    41: {"name": "Tagalog", "code": "tl"},
    42: {"name": "Thai", "code": "th"},
    43: {"name": "Turkish", "code": "tr"},
    44: {"name": "Ukrainian", "code": "uk"},
    45: {"name": "Vietnamese", "code": "vi"},
}

# ============================================================
# LAZY-LOADED IndicBARTSS
# ============================================================
_tokenizer = None
_model = None

def _load_indicbart():
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBARTSS")
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/IndicBARTSS"
    ).to(device)

    _model.eval()

def translate_with_indicbart(text, target_lang_name):
    _load_indicbart()

    prompt = f"translate English to {target_lang_name}: {text}"
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_length=256,
            num_beams=4
        )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_with_argos(text, source_lang, target_lang):
    translation = argostranslate.translate.get_translation_from_codes(
        source_lang, target_lang
    )
    if not translation:
        raise RuntimeError(f"Argos model not found: {source_lang} → {target_lang}")
    return translation.translate(text)

# ============================================================
# PUBLIC ENTRY POINT
# ============================================================
def translate_caption(caption, category, choice):
    if not caption or not caption.strip():
        return {}

    if category == "indic":
        lang = INDIC_LANGUAGES.get(choice)
        translated = translate_with_indicbart(caption, lang["name"])
        return {lang["name"]: translated}

    if category == "international":
        lang = INTERNATIONAL_LANGUAGES.get(choice)
        translated = translate_with_argos(caption, "en", lang["code"])
        return {lang["name"]: translated}

    return {}

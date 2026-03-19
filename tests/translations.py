import os
import sys

# Set HuggingFace cache directory before any imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_HOME'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights', 'hf')

from translation.indic import IndicTransWrapper
from translation.nllb import NLLBWrapper
from translation.router import INDIC_LANGS, NLLB_LANGS, MARIAN_LANGS
from translation.translator import translate_caption


def build_target_languages():
    indic_langs = set(IndicTransWrapper.FLORES_CODES.keys())
    indic_langs.discard("en")

    nllb_langs = set(NLLBWrapper.LANG_MAP.keys())
    nllb_langs.discard("en")

    all_langs = indic_langs | nllb_langs | MARIAN_LANGS
    return sorted(all_langs)


def get_backend(lang_code):
    if lang_code in INDIC_LANGS:
        return "indic-nllb"
    if lang_code in NLLB_LANGS:
        return "nllb"
    return "marian"

def main():
    # Ask for input sentence
    caption = input("Enter the sentence to translate: ")

    # Define target languages (from all backends)
    target_languages = build_target_languages()

    # Generate translations
    translations = translate_caption(caption, target_languages)

    # Define output file path
    output_file_path = os.path.join(os.getcwd(), 'outputs', 'translations_output.txt')

    # Write translations to the output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for lang, translation in translations.items():
            backend = get_backend(lang) if lang != "en" else "source"
            f.write(f"{lang} ({backend}): {translation}\n")

    print(f"Translations saved to {output_file_path}")

if __name__ == "__main__":
    main()
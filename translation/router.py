from .indic import IndicTransWrapper
from .marian import MarianMTWrapper
from .nllb import NLLBWrapper

INDIC_LANGS = {'hi', 'bn', 'ta', 'te', 'kn', 'ml', 'mr', 'gu', 'pa', 'or', 'as'}
NLLB_LANGS = {'ko', 'tr', 'ja', 'pt'}
MARIAN_LANGS = {'ur', 'ar', 'it', 'nl', 'vi', 'id', 'fr', 'es', 'de', 'zh', 'ru'}

def translate_text(text, target_lang, source_lang="en"):
    """
    Routes translation request to the appropriate backend.
    
    Args:
        text (str): Tensor to translate.
        target_lang (str): 2-letter ISO code.
        source_lang (str): 2-letter ISO code (default 'en').
        
    Returns:
        str: Translated text.
    """
    if not text or not text.strip():
        return ""

    try:
        if target_lang in INDIC_LANGS:
            # Use IndicTrans2
            if source_lang != 'en':
                print(f"[WARNING] IndicTrans2 model only supports en->indic. Skipping {source_lang}->{target_lang}")
                return text
                
            translator = IndicTransWrapper.get_instance()
            return translator.translate(text, target_lang, source_lang)
            
        elif target_lang in NLLB_LANGS:
            # Use NLLB
            translator = NLLBWrapper.get_instance()
            return translator.translate(text, target_lang, source_lang)
            
        else:
            # Use MarianMT
            translator = MarianMTWrapper.get_instance(target_lang, source_lang)
            return translator.translate(text)
            
    except Exception as e:
        error_msg = str(e)
        if "sentencepiece" in error_msg.lower():
            print(f"[ERROR] Missing dependency for {target_lang}: 'sentencepiece'. Please run: pip install sentencepiece")
        else:
            print(f"[ERROR] Translation failed for {target_lang}: {e}")
        return text
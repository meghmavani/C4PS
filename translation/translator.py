from .router import translate_text

def translate_caption(caption, target_languages=['fr', 'es', 'de']):
    """
    Translates the caption into multiple target languages using the configured backends.
    
    Args:
        caption (str): The English caption to translate.
        target_languages (list): A list of language codes.
        
    Returns:
        dict: A dictionary mapping language codes to translated text.
              Also includes the original 'en' caption.
    """
    translations = {}
    
    # Always include the original English caption
    translations['en'] = caption
    
    if not caption or not caption.strip():
        print(f"[WARNING] Skipping translation for empty caption.")
        for lang in target_languages:
            if lang != 'en':
                translations[lang] = ""
        return translations

    for target_lang in target_languages:
        if target_lang == 'en': 
            continue
            
        try:
            translated_text = translate_text(caption, target_lang, source_lang="en")
            translations[target_lang] = translated_text
        except Exception as e:
            print(f"[ERROR] Could not translate to {target_lang}: {e}")
            translations[target_lang] = "Translation failed."
            
    return translations
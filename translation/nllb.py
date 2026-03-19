import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class NLLBWrapper:
    """
    Wrapper for NLLB-200 (No Language Left Behind) model.
    Used for languages not supported by MarianMT or IndicTrans2.
    """
    _instance = None

    # Simplified mapping for common codes to NLLB codes (FLORES-200)
    LANG_MAP = {
        'en': 'eng_Latn',
        'ko': 'kor_Hang',
        'tr': 'tur_Latn',
        'fr': 'fra_Latn',
        'es': 'spa_Latn',
        'de': 'deu_Latn',
        'ja': 'jpn_Jpan',
        'zh': 'zho_Hans',
        'ru': 'rus_Cyrl',
        'pt': 'por_Latn',
        # Add others as needed
    }

    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading NLLB model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.lang_map = self.LANG_MAP

    def _get_nllb_code(self, simple_code):
        return self.lang_map.get(simple_code, simple_code)

    def translate(self, text, target_lang, source_lang="en"):
        """
        Translates text using NLLB.
        """
        src_code = self._get_nllb_code(source_lang)
        tgt_code = self._get_nllb_code(target_lang)

        # For NLLB, we set the source language for the tokenizer
        self.tokenizer.src_lang = src_code

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512
            )

        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return result

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # Clean up other heavy models if needed (simple heuristic)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            cls._instance = cls()
        return cls._instance
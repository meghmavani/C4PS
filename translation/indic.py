import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class IndicTransWrapper:
    """
    Wrapper using NLLB-200 (Distilled 600M) for Indian languages.
    Used as a drop-in replacement for IndicTrans2 to avoid gated access issues
    while maintaining high-quality Indic translation support.
    """
    _instance = None
    
    # NLLB / FLORES-200 Language Codes
    FLORES_CODES = {
        "en": "eng_Latn",
        "hi": "hin_Deva",
        "bn": "ben_Beng",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "kn": "kan_Knda",
        "ml": "mal_Mlym",
        "mr": "mar_Deva",
        "gu": "guj_Gujr",
        "pa": "pan_Guru",
        "or": "ory_Orya",
        "as": "asm_Beng"
    }

    def __init__(self):
        # Using the unified NLLB model which supports 200 languages including all Indic ones
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            try:
                import sentencepiece
            except ImportError:
                  raise ImportError("The 'sentencepiece' library is required for NLLB. Please install it with `pip install sentencepiece`.")

            print(f"[INFO] Loading NLLB model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def translate(self, text, target_lang, source_lang="en"):
        self._load_model()
        
        src_code = self.FLORES_CODES.get(source_lang, "eng_Latn")
        tgt_code = self.FLORES_CODES.get(target_lang)
        
        if not tgt_code:
            print(f"[WARNING] Unsupported language code for NLLB: {target_lang}")
            return text

        # NLLB translation logic
        try:
            print(f"[INFO] Translating EN -> {target_lang} using NLLB backend...")
            
            # Set source language
            self.tokenizer.src_lang = src_code
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)

            # Get target language ID - logic that works for NllbTokenizerFast
            # Try convert_tokens_to_ids which is standard
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
            
            # Generate with forced target language
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=512,
                    num_beams=1, # Greedy for speed
                    do_sample=False
                )
            
            result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            if result.strip() == text.strip():
                 print(f"[WARN] NLLB translation output identical to input for {target_lang}")

            return result
            
        except Exception as e:
            print(f"[ERROR] NLLB Translation failed: {e}")
            return text

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
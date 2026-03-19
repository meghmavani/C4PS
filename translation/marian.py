import torch
from transformers import MarianMTModel, MarianTokenizer

class MarianMTWrapper:
    """
    Wrapper for Helsinki-NLP's MarianMT models for non-Indian languages.
    Implements caching to avoid reloading models.
    """
    # Languages routed to MarianMT (ISO 639-1 codes)
    LANGS = {
        "ur", "ar", "it", "nl", "vi", "id",
        "fr", "es", "de", "zh", "ru"
    }

    # Cache only the single most recent instance to prevent OOM
    # on constrained hardware (GTX 1650 4GB).
    _current_instance = None
    _current_key = None

    def __init__(self, target_lang, source_lang="en"):
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            print(f"[INFO] Loading MarianMT model: {self.model_name}...")
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def translate(self, text):
        self._load_model()
        
        # Prepare input
        encoded_text = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        # Generate translation (deterministic)
        with torch.no_grad():
            gen_tokens = self.model.generate(
                **encoded_text,
                max_new_tokens=512,
                do_sample=False,  # Deterministic
                num_beams=1       # Greedy search
            )
            
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    @classmethod
    def get_instance(cls, target_lang, source_lang="en"):
        key = f"{source_lang}-{target_lang}"
        
        # If we already have this model loaded, return it
        if cls._current_key == key and cls._current_instance is not None:
            return cls._current_instance
            
        # Othewise, unload previous model if exists to save memory
        if cls._current_instance is not None:
            print(f"[INFO] Unloading MarianMT model for {cls._current_key} to free memory...")
            # Delete model and tokenizer from the instance
            if cls._current_instance.model is not None:
                del cls._current_instance.model
            if cls._current_instance.tokenizer is not None:
                del cls._current_instance.tokenizer
            del cls._current_instance
            
            # Force garbage collection and CUDA cache clear
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        # Create new instance
        cls._current_instance = cls(target_lang, source_lang)
        cls._current_key = key
        return cls._current_instance
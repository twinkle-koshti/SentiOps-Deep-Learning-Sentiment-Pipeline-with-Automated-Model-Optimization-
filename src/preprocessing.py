import re
from transformers import AutoTokenizer

class SentiPreprocessor:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_pad(self, text_list, max_length=128):
        cleaned_texts = [self.clean_text(t) for t in text_list]
        encoded_inputs = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encoded_inputs

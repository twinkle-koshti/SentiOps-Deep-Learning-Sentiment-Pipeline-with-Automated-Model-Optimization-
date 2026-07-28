import time
import os
import psutil
import torch
from transformers import AutoModelForSequenceClassification

class SentiEngine:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def get_ram_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def predict_sentiment(self, encoded_inputs):
        start_time = time.time()
        start_ram = self.get_ram_usage()

        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        end_time = time.time()
        end_ram = self.get_ram_usage()

        latency = (end_time - start_time)
        ram_consumed = max(0.1, end_ram - start_ram)

        results = []
        for pred in predictions:
            neg_score = pred[0].item()
            pos_score = pred[1].item()
            
            if pos_score > neg_score:
                label = "POSITIVE"
                confidence = pos_score * 100
            else:
                label = "NEGATIVE"
                confidence = neg_score * 100
            results.append({"label": label, "confidence": round(confidence, 2)})

        return results, latency, ram_consumed

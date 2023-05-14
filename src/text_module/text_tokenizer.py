from typing import Dict, List
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoTokenizer

class Tokenizer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            text=texts,
            padding=self.config["tokenizer"]["padding"],
            max_length=self.config["tokenizer"]["max_length"],
            truncation=self.config["tokenizer"]["truncation"],
            return_tensors='pt',
            return_token_type_ids=self.config["tokenizer"]["return_token_type_ids"],
            return_attention_mask=self.config["tokenizer"]["return_attention_mask"],
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
            "pad_token_id": encoded_text['pad_token_id'].squeeze(),
        }
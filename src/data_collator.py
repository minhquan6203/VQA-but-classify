import os
from dataclasses import dataclass
from typing import Dict, List
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoFeatureExtractor

@dataclass
class MultimodalCollator:
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
        self.preprocessor = AutoFeatureExtractor.from_pretrained(config["model"]["image_encoder"])

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
        }

    def preprocess_images(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[
                Image.open(
                    os.path.join(
                        self.config["data"]["dataset_folder"],
                        self.config["data"]["images_folder"], 
                        image
                    )
                ).convert('RGB') for image in images
            ],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }
            
    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text([ann["question"] for raw in raw_batch_dict for ann in raw["annotations"]]

            ),
            **self.preprocess_images([ann["filename"] for raw in raw_batch_dict for ann in raw["images"]]),
            'labels': torch.tensor([raw_batch_dict['answer_space'].index(ann["answers"][0]) for raw in raw_batch_dict for ann in raw["annotations"]],
                dtype=torch.int64
            ),

        }

def createMultimodalDataCollator(config: Dict) -> MultimodalCollator:
    collator = MultimodalCollator(config)
    return collator

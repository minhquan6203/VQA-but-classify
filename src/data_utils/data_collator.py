import os
from dataclasses import dataclass
from typing import Dict, List
from PIL import Image

import torch
from vision_module.vision_preprocess import Processcer
from text_module.text_tokenizer import Tokenizer

@dataclass
class MultimodalCollator:
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = Tokenizer(config)
        self.preprocessor = Processcer(config)
            
    def __call__(self, raw_batch_dict):
        return {
            **self.tokenizer.tokenize_text([ann["question"] for ann in raw_batch_dict]),
            **self.preprocessor.preprocess_images([ann["image_id"] for ann in raw_batch_dict]),
            'labels': torch.tensor([i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),

        }

def createMultimodalDataCollator(config: Dict) -> MultimodalCollator:
    collator = MultimodalCollator(config)
    return collator

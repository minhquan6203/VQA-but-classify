import os
from dataclasses import dataclass
from typing import Dict, List
from PIL import Image
import torch

@dataclass
class MultimodalCollator:
    def __init__(self, config: Dict):
        self.config = config
    def __call__(self, raw_batch_dict):
        return {
           'questions':[ann["question"] for ann in raw_batch_dict],
            'images': [ann["image_id"] for ann in raw_batch_dict],
            'labels': torch.tensor([i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),

        }

def createDataCollator(config: Dict) -> MultimodalCollator:
    collator = MultimodalCollator(config)
    return collator

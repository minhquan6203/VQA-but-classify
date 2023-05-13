from PIL import Image
import os
from typing import Dict, List
from PIL import Image
import torch
from transformers import AutoFeatureExtractor

class Processcer:
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.preprocessor = AutoFeatureExtractor.from_pretrained(config["vision_embedding"]["image_encoder"])

    def preprocess_images(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[
                Image.open(
                    os.path.join(
                        self.config["data"]["dataset_folder"],
                        self.config["data"]["images_folder"], 
                        str(image_id).zfill(12) + ".jpg"
                    )
                ).convert('RGB') for image_id in images
            ],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }
            
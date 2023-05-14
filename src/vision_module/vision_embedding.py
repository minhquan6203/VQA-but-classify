import torch
from torch import nn
import os
from PIL import Image
from typing import List
from transformers import AutoModel, AutoFeatureExtractor
from typing import List, Dict, Optional
from mask.masking import generate_padding_mask

class Vision_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super(Vision_Embedding,self).__init__()
        self.backbone = AutoModel.from_pretrained(config["vision_embedding"]["image_encoder"])
        self.preprocessor = AutoFeatureExtractor.from_pretrained(config["vision_embedding"]["image_encoder"])
        # freeze all parameters of pretrained model
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.proj = nn.Linear(config["vision_embedding"]['d_features'], config["vision_embedding"]['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["text_embedding"]['dropout'])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_folder = config["data"]["dataset_folder"]
        self.image_folder = config["data"]["images_folder"]
    def forward(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[
                Image.open(os.path.join(self.data_folder,self.image_folder, str(image_id).zfill(12) + ".jpg")
                ).convert('RGB') for image_id in images
            ],
            return_tensors="pt",
        ).to(self.device)
        features = self.backbone(**processed_images).last_hidden_state
        padding_mask = generate_padding_mask(features, padding_idx=0)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask
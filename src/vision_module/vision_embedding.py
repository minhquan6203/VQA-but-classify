import torch
from torch import nn
from PIL import Image
from typing import List
from transformers import AutoModel
from typing import List, Dict, Optional
from mask.masking import generate_padding_mask

class Vision_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(config["vision_embedding"]["image_encoder"])
        # freeze all parameters of pretrained model
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.proj = nn.Linear(config["vision_embedding"]['d_features'], config["vision_embedding"]['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["text_embedding"]['dropout'])

    def forward(self, pixel_values):
        features = self.backbone(pixel_values,return_dict=True)
        padding_mask = generate_padding_mask(features, padding_idx=0)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask
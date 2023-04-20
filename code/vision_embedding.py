import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel
from PIL import Image
from typing import List
from utils import generate_padding_mask


class Vision_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = torch.device(config.DEVICE)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.VISION_EMBEDDING.PRETRAINED_NAME)
        self.backbone = AutoModel.from_pretrained(config.VISION_EMBEDDING.PRETRAINED_NAME)
        # freeze all parameters of pretrained model
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.proj = nn.Linear(config.VISION_EMBEDDING.D_FEATURE, config.VISION_EMBEDDING.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.VISION_EMBEDDING.DROPOUT)

    def forward(self, images: List[Image.Image]):
        inputs = self.feature_extractor(images, return_tensors="pt").to(self.device)
        features = self.backbone(**inputs).last_hidden_state
        padding_mask = generate_padding_mask(features, padding_idx=0)

        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        
        return out, padding_mask
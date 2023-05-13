import torch
from torch import nn
from torch.nn import functional as F
from transformers import  AutoModel
from typing import List, Dict, Optional
from mask.masking import generate_padding_mask


class Text_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.embedding = AutoModel.from_pretrained(config["text_emmbedding"]["text_encoder"])
        # freeze all parameters of pretrained model
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.proj = nn.Linear(config["text_emmbedding"]['d_feature'], config["text_emmbedding"]['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["text_emmbedding"]['dropout'])

    def forward(self, input_ids, attention_mask, token_type_ids, pad_token_id):
        
        features = self.embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        padding_mask = generate_padding_mask(input_ids, padding_idx=pad_token_id)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask

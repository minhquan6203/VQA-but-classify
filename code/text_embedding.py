import torch
from torch import nn
from torch.nn import functional as F

from utils import generate_padding_mask

from transformers import AutoTokenizer, AutoModel
from typing import Dict, List


class Text_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config.DEVICE

        self.tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_NAME)
        self.embedding = AutoModel.from_pretrained(config.PRETRAINED_NAME)
        # freeze all parameters of pretrained model
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.proj = nn.Linear(config.D_PRETRAINED_FEATURE, config.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions, return_tensors="pt", padding=True).to(self.device)
        padding_mask = generate_padding_mask(inputs.input_ids, padding_idx=self.tokenizer.pad_token_id)
        features = self.embedding(**inputs).last_hidden_state

        out = self.proj(features)
        out = self.dropout(self.gelu(out))

        return out, padding_mask

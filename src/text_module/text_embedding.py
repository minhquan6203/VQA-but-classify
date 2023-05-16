import torch
from torch import nn
from torch.nn import functional as F
from transformers import  AutoModel, AutoTokenizer
from typing import List, Dict, Optional
from mask.masking import generate_padding_mask
from text_module.tokenizer import Text_Tokenizer


class Text_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super(Text_Embedding,self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        self.embedding = AutoModel.from_pretrained(config["text_embedding"]["text_encoder"])
        # freeze all parameters of pretrained model
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.proj = nn.Linear(config["text_embedding"]['d_features'], config["text_embedding"]['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["text_embedding"]['dropout'])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = Text_Tokenizer(config)

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(text = questions).to(self.device)
        features = self.embedding(**inputs).last_hidden_state
        padding_mask = generate_padding_mask(inputs.input_ids, padding_idx=self.tokenizer.pad_token_id)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask

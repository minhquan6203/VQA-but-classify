import torch
from torch import nn
from torch.nn import functional as F
from transformers import  AutoModel, AutoTokenizer
from typing import List, Dict, Optional
from mask.masking import generate_padding_mask


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

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(
            text=questions,
            padding=self.config["tokenizer"]["padding"],
            max_length=self.config["tokenizer"]["max_length"],
            truncation=self.config["tokenizer"]["truncation"],
            return_tensors='pt',
            return_token_type_ids=self.config["tokenizer"]["return_token_type_ids"],
            return_attention_mask=self.config["tokenizer"]["return_attention_mask"],
        ).to(self.device)
        features = self.embedding(**inputs).last_hidden_state

        padding_mask = generate_padding_mask(inputs.input_ids, padding_idx=self.tokenizer.pad_token_id)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask

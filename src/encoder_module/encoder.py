import torch
from torch import nn
from typing import List, Dict, Optional

from utils.positional_feed_forward import PositionWiseFeedForward
from attention_module.attentions import MultiHeadAtt
from utils.positional_embbeding import SinusoidPositionalEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, config: Dict):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAtt(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        att = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask, **kwargs)
        ff = self.pwff(att)

        return ff

class CoAtt_Encoder(nn.Module):
    def __init__(self, config: Dict):
        super(CoAtt_Encoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(config["encoder"]['d_model'])
        self.layer_norm = nn.LayerNorm(config["encoder"]['d_model'])
        self.matt = MultiHeadAtt(config)
        self.attention_weights = nn.Linear(config["encoder"]['d_model'], 1)  # Khai báo lớp nn.Linear cho attention_weights

    def forward(self,text_feature, text_mask, image_feature, image_mask):
        image_feature = self.layer_norm(image_feature) + self.pos_embedding(image_feature)
        text_feature = self.layer_norm(text_feature) + self.pos_embedding(text_feature)
        # Multi-Head Attention
        text_attended = self.matt(text_feature, text_feature, text_feature, text_mask)
        image_attended = self.matt(image_feature, image_feature, image_feature, image_mask)

        # Cross attention
        text_attended = self.matt(text_attended, image_attended, image_attended, image_mask)
        image_attended = self.matt(image_attended, text_attended, text_attended, text_mask)

        # Co-attention
        text_attended = self.attention_weights(torch.tanh(text_attended))
        image_attended = self.attention_weights(torch.tanh(image_attended))
        attention_weights = torch.softmax(torch.cat([text_attended, image_attended], dim=1), dim=1)
        attended_text = torch.sum(attention_weights[:, 0].unsqueeze(-1) * text_feature, dim=1)
        attended_image = torch.sum(attention_weights[:, 1].unsqueeze(-1) * image_feature, dim=1)
        
        return attended_text, attended_image


# import torch
# from torch import nn

# from utils.positional_feed_forward import PositionWiseFeedForward
# from attention_module.attentions import MultiHeadAtt
# from utils.positional_embbeding import SinusoidPositionalEmbedding

# class EncoderLayer(nn.Module):
#     def __init__(self, config):
#         super(EncoderLayer, self).__init__()
#         self.mhatt = MultiHeadAtt(config)
#         self.pwff = PositionWiseFeedForward(config)

#     def forward(self, queries, keys, values, attention_mask, **kwargs):
#         att = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask)
#         ff = self.pwff(att)

#         return ff


# class CoAttentionEncoder(nn.Module):
#     '''
#         This module is designed inspired from ViLBERT (https://arxiv.org/pdf/1908.02265.pdf).
#     '''
#     def __init__(self, config):
#         super(CoAttentionEncoder, self).__init__()

#         self.pos_embedding = SinusoidPositionalEmbedding(config["encoder"]['d_model'])
#         self.vision_layer_norm = nn.LayerNorm(config["encoder"]['d_model'])
#         self.language_layer_norm = nn.LayerNorm(config["encoder"]['d_model'])

#         self.d_model = config["encoder"]['d_model']

#         # cross-attention layers
#         self.vision_language_attn_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config["encoder"]['layers'])])
#         self.language_vision_attn_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config["encoder"]['layers'])])

#         # self-attention layers
#         self.vision_self_attn_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config["encoder"]['layers'])])
#         self.language_self_attn_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config["encoder"]['layers'])])

#     def forward(self, vision_features: torch.Tensor, vision_padding_mask: torch.Tensor, 
#                 language_features: torch.Tensor, language_padding_mask: torch.Tensor):
#         vision_features = self.vision_layer_norm(vision_features) + self.pos_embedding(vision_features)
#         language_features = self.language_layer_norm(language_features) + self.pos_embedding(language_features)
#         for layers in zip(self.vision_language_attn_layers, 
#                             self.language_vision_attn_layers, 
#                             self.vision_self_attn_layers, 
#                             self.language_self_attn_layers):
#             vision_language_attn_layer, language_vision_attn_layer, vision_self_attn_layer, language_self_attn_layer = layers
#             # performing cross-attention
#             vision_features = vision_language_attn_layer(
#                 queries=vision_features,
#                 keys=language_features,
#                 values=language_features,
#                 attention_mask=language_padding_mask
#             )
#             language_features = language_vision_attn_layer(
#                 queries=language_features,
#                 keys=vision_features,
#                 values=vision_features,
#                 attention_mask=vision_padding_mask
#             )
#             # performing self-attention
#             vision_features = vision_self_attn_layer(
#                 queries=vision_features,
#                 keys=vision_features,
#                 values=vision_features,
#                 attention_mask=vision_padding_mask
#             )
#             language_features = language_self_attn_layer(
#                 queries=language_features,
#                 keys=language_features,
#                 values=language_features,
#                 attention_mask=language_padding_mask
#             )

#         return vision_features, language_features


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
        self.encoder_layer = EncoderLayer(config)
        self.attention_weights = nn.Linear(config["encoder"]['d_model'], 1)  # Khai báo lớp nn.Linear cho attention_weights

    def forward(self,text_feature, text_mask, image_feature, image_mask):
        image_feature = self.layer_norm(image_feature)
        text_feature = self.layer_norm(text_feature)
        # Multi-Head Attention
        text_attended = self.encoder_layer(text_feature, text_feature, text_feature, text_mask)
        image_attended = self.encoder_layer(image_feature, image_feature, image_feature, image_mask)

        # Cross attention
        text_attended = self.encoder_layer(text_attended, image_attended, image_attended, image_mask)
        image_attended = self.encoder_layer(image_attended, text_attended, text_attended, text_mask)

        # Co-attention
        text_attended = self.attention_weights(torch.tanh(text_attended))
        image_attended = self.attention_weights(torch.tanh(image_attended))
        attention_weights = torch.softmax(torch.cat([text_attended, image_attended], dim=1), dim=1)
        attended_text = torch.sum(attention_weights[:, 0].unsqueeze(-1) * text_feature, dim=1)
        attended_image = torch.sum(attention_weights[:, 1].unsqueeze(-1) * image_feature, dim=1)
        
        return attended_text, attended_image


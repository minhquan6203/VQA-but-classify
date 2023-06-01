import torch
from torch import nn
from typing import List, Dict, Optional
from utils.positional_feed_forward import PositionWiseFeedForward
from attention_module.attentions import MultiHeadAtt
from utils.positional_embbeding import SinusoidPositionalEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.mhatt = MultiHeadAtt(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, attention_mask, **kwargs):
        att = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask)
        ff = self.pwff(att)

        return ff


class CoAttentionEncoder(nn.Module):
    '''
        This module is designed inspired from ViLBERT (https://arxiv.org/pdf/1908.02265.pdf).
    '''
    def __init__(self, config):
        super(CoAttentionEncoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(config["encoder"]['d_model'])
        self.vision_layer_norm = nn.LayerNorm(config["encoder"]['d_model'])
        self.language_layer_norm = nn.LayerNorm(config["encoder"]['d_model'])

        self.d_model = config["encoder"]['d_model']

        # cross-attention layers
        self.vision_language_attn_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config["encoder"]['layers'])])
        self.language_vision_attn_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config["encoder"]['layers'])])

        # self-attention layers
        self.vision_self_attn_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config["encoder"]['layers'])])
        self.language_self_attn_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config["encoder"]['layers'])])

    def forward(self, vision_features: torch.Tensor, vision_padding_mask: torch.Tensor, 
                language_features: torch.Tensor, language_padding_mask: torch.Tensor):
        vision_features = self.vision_layer_norm(vision_features) + self.pos_embedding(vision_features)
        language_features = self.language_layer_norm(language_features) + self.pos_embedding(language_features)
        for layers in zip(self.vision_language_attn_layers, 
                            self.language_vision_attn_layers, 
                            self.vision_self_attn_layers, 
                            self.language_self_attn_layers):
            vision_language_attn_layer, language_vision_attn_layer, vision_self_attn_layer, language_self_attn_layer = layers
            # performing cross-attention
            vision_features = vision_language_attn_layer(
                queries=vision_features,
                keys=language_features,
                values=language_features,
                attention_mask=language_padding_mask
            )
            language_features = language_vision_attn_layer(
                queries=language_features,
                keys=vision_features,
                values=vision_features,
                attention_mask=vision_padding_mask
            )
            # performing self-attention
            vision_features = vision_self_attn_layer(
                queries=vision_features,
                keys=vision_features,
                values=vision_features,
                attention_mask=vision_padding_mask
            )
            language_features = language_self_attn_layer(
                queries=language_features,
                keys=language_features,
                values=language_features,
                attention_mask=language_padding_mask
            )

        return vision_features, language_features


#encoder for mcan
class GuidedEncoderLayer(nn.Module):
    def __init__(self, config):
        super(GuidedEncoderLayer, self).__init__()
        self.self_mhatt = MultiHeadAtt(config)
        self.guided_mhatt = MultiHeadAtt(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, self_attention_mask, guided_attention_mask, **kwargs):
        self_att = self.self_mhatt(queries=queries,keys=queries, values=queries,attention_mask=self_attention_mask,**kwargs)
        guided_att = self.guided_mhatt(queries=self_att, keys=keys, values=values,attention_mask=guided_attention_mask,**kwargs)

        ff = self.pwff(guided_att)

        return ff
    

class GuidedAttentionEncoder(nn.Module):
    '''
        This module is designed inspired from Deep Modular Co-Attention Network (https://arxiv.org/pdf/1906.10770.pdf).
    '''
    def __init__(self, config):
        super(GuidedAttentionEncoder, self).__init__()

        self.pos_embedding = SinusoidPositionalEmbedding(config["encoder"]['d_model'])
        self.layer_norm = nn.LayerNorm(config["encoder"]['d_model'])
        self.d_model = config["encoder"]['d_model']

        self.guided_attn_layers = nn.ModuleList([GuidedEncoderLayer(config) for _ in range(config["encoder"]['layers'])])
        self.language_self_attn_layers = nn.ModuleList([GuidedEncoderLayer(config) for _ in range(config["encoder"]['layers'])])

    def forward(self, vision_features: torch.Tensor, vision_padding_mask: torch.Tensor, 
                language_features: torch.Tensor, language_padding_mask: torch.Tensor):
        vision_features = self.layer_norm(vision_features) + self.pos_embedding(vision_features)
        language_features = self.layer_norm(language_features) + self.pos_embedding(language_features)

        for layer in zip(self.guided_attn_layers,self.language_self_attn_layers):
            guided_attn_layer,language_self_attn_layers=layer
            vision_features = guided_attn_layer(
                queries=vision_features,
                keys=language_features,
                values=language_features,
                self_attention_mask=vision_padding_mask,
                guided_attention_mask=language_padding_mask
            )
            language_features=language_self_attn_layers(
                queries=language_features,
                keys=language_features,
                values=language_features,
                attention_mask=language_padding_mask
            )
        

        return vision_features,language_features

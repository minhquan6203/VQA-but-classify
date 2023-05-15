import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Dict, Optional
from attention_module.attentions import MultiHeadAtt
from mask.masking import generate_padding_mask, generate_sequential_mask, generate_self_attention_masks, sinusoid_encoding_table
from utils.positional_feed_forward import PositionWiseFeedForward
from text_module.text_embedding import Text_Embedding
from transformers import AutoTokenizer, BertGenerationDecoder


class DecoderLayer(nn.Module):
    def __init__(self, config: Dict):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAtt(config)
        self.enc_attn = MultiHeadAtt(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, self_attention_mask, enc_attention_mask, **kwargs):
        self_att = self.self_attn(queries, queries, queries, attention_mask=self_attention_mask, **kwargs)
        enc_att = self.enc_attn(self_att, keys, values, attention_mask=enc_attention_mask, **kwargs)

        ff = self.pwff(enc_att)
        
        return ff


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.d_model = config['decoder']['d_model']
        self.N = config['decoder']['layers']
        self.max_len = config['decoder']['max_len']
        self.gen = BertGenerationDecoder.from_pretrained(config['decoder']['text_decoder'])

  
    def forward(self, answer_tokens: torch.Tensor, encoder_features: torch.Tensor, encoder_attention_mask: torch.Tensor):

        out = self.gen(inputs_embeds=encoder_features,attention_mask=encoder_attention_mask,labels=answer_tokens)
        
        return out.logits, out.loss
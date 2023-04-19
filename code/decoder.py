import torch
from torch import nn
from torch.nn import functional as F

from attention import MultiHeadAttention
from utils import generate_padding_mask, generate_sequential_mask, generate_self_attention_masks, sinusoid_encoding_table
from positionwise_feed_forward import PositionWiseFeedForward
from language_model import Language_Model
from text_embedding import Text_Embedding


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(config)
        self.enc_attn = MultiHeadAttention(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, self_attention_mask, enc_attention_mask, **kwargs):
        self_att = self.self_attn(queries, queries, queries, attention_mask=self_attention_mask, **kwargs)
        enc_att = self.enc_attn(self_att, keys, values, attention_mask=enc_attention_mask, **kwargs)

        ff = self.pwff(enc_att)
        
        return ff

class Decoder(nn.Module):
    def __init__(self, config, vocab):
        super(Decoder, self).__init__()

        self.d_model = config.DECODER.D_MODEL
        self.max_len = vocab.max_answer_length
        self.padding_idx = vocab.padding_idx
        self.N = config.DECODER.LAYERS

        self.word_emb = Text_Embedding(config)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len=self.max_len+1,
                                                                            d_model=config.DECODER.D_MODEL, padding_idx=0), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(config) if i < config.DECODER.LAYERS else DecoderLayer(config) 
                                                for i in range(config.DECODER.LAYERS + 1)])
        self.fc = nn.Linear(config.DECODER.D_MODEL, len(vocab), bias=False)

        # load and froze the language model
        self.language_model = Language_Model(config.DECODER.LANGUAGE_MODEL)

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, answer_tokens: torch.Tensor, encoder_features: torch.Tensor, encoder_attention_mask: torch.Tensor):
        b_s, seq_len = answer_tokens.shape
        answer_padding_masks = generate_padding_mask(answer_tokens, self.padding_idx).to(answer_tokens.device)
        answer_self_attention_masks = generate_sequential_mask(seq_len).to(answer_tokens.device)
        answer_self_attention_masks = generate_self_attention_masks(answer_padding_masks, answer_self_attention_masks)
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, answer_self_attention_masks], -1)
            answer_self_attention_masks = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(answer_tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(answer_padding_masks != 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        # get the language_signals
        _, language_signals = self.language_model(answer_tokens)

        embedded_answers, _ = self.word_emb(answer_tokens)
        out = embedded_answers + self.pos_emb(seq)
        for layer in self.layers:
            out = layer(queries=out, 
                        keys=encoder_features,
                        values=encoder_features,
                        language_signals=language_signals,
                        self_attention_mask=answer_self_attention_masks,
                        enc_attention_mask=encoder_attention_mask)

        out = self.fc(out)

        return F.log_softmax(out, dim=-1)
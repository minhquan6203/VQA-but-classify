import torch
from torch import nn
from torch.nn import functional as F

from vision_embedding import Vision_Embedding
from text_embedding import Text_Embedding
from instance import Instance
from decoder import Decoder
from encoder import CoAttentionEncoder
from beam_search import BeamSearch

class BaseTransformer(nn.Module):
    def __init__(self, config, vocab):
        super(BaseTransformer, self).__init__()

        self.vocab = vocab
        self.max_len = vocab.max_answer_length
        self.eos_idx = vocab.eos_idx
        self.d_model = config.MODEL.D_MODEL


    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encoder_forward(self, input_features: Instance):
        raise NotImplementedError

    def forward(self, input_features: Instance):
        raise NotImplementedError

    def step(self, t, prev_output):
        bs = self.encoder_features.shape[0]
        if t == 0:
            it = torch.zeros((bs, 1)).long().fill_(self.vocab.bos_idx).to(self.encoder_features.device)
        else:
            it = prev_output

        output = self.decoder(
            answer_tokens=it,
            encoder_features=self.encoder_features,
            encoder_attention_mask=self.encoder_padding_mask
        )

        return output

    def beam_search(self, input_features: Instance, batch_size: int, beam_size: int, out_size=1, return_probs=False, **kwargs):
        beam_search = BeamSearch(model=self, max_len=self.max_len, eos_idx=self.eos_idx, beam_size=beam_size, 
                            b_s=batch_size, device=self.device)

        with self.statefulness(batch_size):
            self.encoder_features, self.encoder_padding_mask = self.encoder_forward(input_features)
            output =  beam_search.apply(out_size, return_probs, **kwargs)

        return output

class ViTmBERTGeneration(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)

        self.vision_embedding = Vision_Embedding(config)
        self.text_embedding = Text_Embedding(config)

        self.fusion = nn.Linear(config.MODEL.D_MODEL, config.MODEL.D_MODEL)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.MODEL.DROPOUT)
        self.norm = nn.LayerNorm(config.MODEL.D_MODEL)

        self.decoder = Decoder(config, vocab)
        self.encoder = CoAttentionEncoder(config)

    def forward(self, inputs: Instance):
        fused_features, fused_padding_mask = self.encoder_forward(inputs)
        
        answer_tokens = inputs.answer_tokens
        out = self.decoder(
            answer_tokens=answer_tokens,
            encoder_features=fused_features,
            encoder_attention_mask=fused_padding_mask
        )

        return F.log_softmax(out, dim=-1)

    def encoder_forward(self, inputs: Instance):
        images = inputs.images
        questions = inputs.question

        vision_features, vision_padding_mask = self.vision_embedding(images)
        text_features, text_padding_mask = self.text_embedding(questions)
        vision_features,text_features=self.encoder(vision_features,vision_padding_mask,text_features,text_padding_mask)
        fused_features = torch.cat([vision_features, text_features], dim=1)
        fused_features = self.gelu(self.fusion(fused_features))
        fused_features = self.dropout(fused_features)
        fused_padding_mask = torch.cat([vision_padding_mask, text_padding_mask], dim=-1)
        return fused_features, fused_padding_mask
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.text_embedding import Text_Embedding
from vision_module.vision_embedding import  Vision_Embedding
from attention_module.attentions import MultiHeadAtt
from encoder_module.encoder import CoAttentionEncoder
from decoder_module.decoder import Decoder
from transformers import AutoTokenizer

#lấy ý tưởng từ MCAN
class MultimodalVQAModel(nn.Module):
    def __init__(self,config: Dict, answer_space):
     
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = len(answer_space)
        self.answer_space = answer_space
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.num_attention_heads=config["attention"]['heads']
        self.d_text = config["text_embedding"]['d_features']
        self.d_vision = config["vision_embedding"]['d_features']
        self.text_embbeding = Text_Embedding(config)
        self.vision_embbeding = Vision_Embedding(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        self.decoder = Decoder(config)
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.encoder = CoAttentionEncoder(config)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.CrossEntropyLoss()
        self.linear = nn.Linear(self.intermediate_dims,768)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, questions: List[str], images: List[str], labels: Optional[torch.LongTensor] = None):
        answers = [self.answer_space[label] for label in labels]
        embbed_text, text_mask= self.text_embbeding(questions)
        embbed_vision, vison_mask = self.vision_embbeding(images)
        encoded_text, encoded_image = self.encoder(embbed_text, text_mask, embbed_vision, vison_mask)
        # text_attended = self.attention_weights(torch.tanh(encoded_text))
        # image_attended = self.attention_weights(torch.tanh(encoded_image))
        # attention_weights = torch.softmax(torch.cat([text_attended, image_attended], dim=1), dim=1)
        # attended_text = torch.sum(attention_weights[:, 0].unsqueeze(-1) * encoded_text, dim=1)
        # attended_image = torch.sum(attention_weights[:, 1].unsqueeze(-1) * encoded_image, dim=1)
        
        fused_output = self.fusion(torch.cat([encoded_text, encoded_image], dim=1))
        fused_output = self.linear(fused_output)
        answers = self.tokenizer.batch_encode_plus(answers,padding='max_length',truncation=True,max_length=fused_output.shape[1],return_tensors='pt').to(self.device)
        answers_ids = answers['input_ids']
        answers_mask = answers['attention_mask']
        fused_mask = self.fusion(torch.cat([text_mask.squeeze(1).squeeze(1),vison_mask.squeeze(1).squeeze(1)],dim=1))
        logits,loss = self.decoder(answers_ids, fused_output, fused_mask)
        out = {
            "logits": logits,
            "loss": loss
        }
     
        return out

def createMultimodalModelForVQA(config: Dict, answer_space: List[str]) -> MultimodalVQAModel:
    model = MultimodalVQAModel(config, answer_space)
    return model
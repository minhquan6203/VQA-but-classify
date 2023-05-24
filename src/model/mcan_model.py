from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.text_embedding import Text_Embedding
from vision_module.vision_embedding import  Vision_Embedding
from attention_module.attentions import MultiHeadAtt
from encoder_module.encoder import GuidedAttentionEncoder
#lấy ý tưởng từ MCAN
class MultimodalVQAModel(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.num_attention_heads=config["attention"]['heads']
        self.d_text = config["text_embedding"]['d_features']
        self.d_vision = config["vision_embedding"]['d_features']
        self.text_embbeding = Text_Embedding(config)
        self.vision_embbeding = Vision_Embedding(config)
        self.matt = MultiHeadAtt(config)
        self.fusion = nn.Sequential(
            nn.Linear(self.intermediate_dims +self.intermediate_dims, self.intermediate_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.encoder = GuidedAttentionEncoder(config)
        self.classifier = nn.Linear(self.intermediate_dims, self.num_labels)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, questions: List[str], images: List[str], labels: Optional[torch.LongTensor] = None):
        embbed_text, text_mask= self.text_embbeding(questions)
        embbed_vision, vison_mask = self.vision_embbeding(images)
        encoded_text, encoded_image = self.encoder(embbed_text, text_mask, embbed_vision, vison_mask)
        
        text_attended = self.attention_weights(torch.tanh(encoded_text))
        image_attended = self.attention_weights(torch.tanh(encoded_image))
        attention_weights = torch.softmax(torch.cat([text_attended, image_attended], dim=1), dim=1)
        attended_text = torch.sum(attention_weights[:, 0].unsqueeze(-1) * encoded_text, dim=1)
        attended_image = torch.sum(attention_weights[:, 1].unsqueeze(-1) * encoded_image, dim=1)
        
        fused_output = self.fusion(torch.cat([attended_text, attended_image], dim=1))
        logits = self.classifier(fused_output)
        logits = F.log_softmax(logits, dim=-1)
        out = {
            "logits": logits
        }
        if labels is not None:
            # logits=logits.view(-1,self.num_labels)
            # labels = labels.view(-1)
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out


def createMultimodalModelForVQA(config: Dict, answer_space: List[str]) -> MultimodalVQAModel:
    model = MultimodalVQAModel(config, num_labels=len(answer_space))
    return model
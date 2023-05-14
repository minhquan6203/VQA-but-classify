from typing import List, Dict, Optional
import torch
from torch import nn
from transformers import AutoModel

#model này lấy ý tưởng từ mcan
class MultimodalVQAModel(nn.Module):
    def __init__(
            self,
            num_labels: int,
            intermediate_dims: int,
            dropout: float,
            pretrained_text_name: str,
            pretrained_image_name: str):
     
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name
        self.intermediate_dims=intermediate_dims
        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name,
        )
        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name,
        )
        self.text_attention = nn.Linear(self.text_encoder.config.hidden_size, self.intermediate_dims)
        self.image_attention = nn.Linear(self.image_encoder.config.hidden_size, self.intermediate_dims)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, self.intermediate_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self.intermediate_dims, self.num_labels)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None):
        
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )

        # Text attention
        text_hidden_states = encoded_text['last_hidden_state']
        text_att = self.text_attention(text_hidden_states)
        text_att = torch.tanh(text_att)
        text_att = self.attention_weights(text_att)
        text_att = torch.softmax(text_att, dim=1)
        text_weighted = torch.bmm(text_att.transpose(1, 2), text_hidden_states)

        # Image attention
        image_hidden_states = encoded_image['last_hidden_state']
        image_att = self.image_attention(image_hidden_states)
        image_att = torch.tanh(image_att)
        image_att = self.attention_weights(image_att)
        image_att = torch.softmax(image_att, dim=1)
        image_weighted = torch.bmm(image_att.transpose(1, 2), image_hidden_states)

        # Co-attention
        attention_weights = torch.bmm(text_weighted, image_weighted.transpose(1, 2))
        attention_weights = torch.softmax(attention_weights, dim=2)
        attended_text = torch.bmm(attention_weights, image_weighted)
        attended_image = torch.bmm(attention_weights.transpose(1, 2), text_weighted)

        # Fusion and classification
        fused_output = self.fusion(torch.cat([attended_text.squeeze(), attended_image.squeeze()], dim=1))
        logits = self.classifier(fused_output)

        out = {
            "logits": logits
        }

        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out



def createMultimodalModelForVQA(config: Dict, answer_space: List[str]) -> MultimodalVQAModel:
    model = MultimodalVQAModel(
        num_labels=len(answer_space),
        intermediate_dims=config["model"]["intermediate_dims"],
        dropout=config["model"]["dropout"],
        pretrained_text_name=config["model"]["text_encoder"],
        pretrained_image_name=config["model"]["image_encoder"]
    )

    return model
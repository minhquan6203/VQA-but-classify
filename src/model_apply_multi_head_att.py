from typing import List, Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoModel

#lấy ý tưởng từ MCAN, apply multi head attention
class MultimodalVQAModel(nn.Module):
    def __init__(
            self,
            num_labels: int,
            intermediate_dims: int,
            dropout: float,
            pretrained_text_name: str,
            pretrained_image_name: str,
            num_attention_heads: int):
     
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name
        self.intermediate_dims=intermediate_dims
        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name,
        )
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name,
        )
        for param in self.image_encoder.parameters():
            param.requires_grad = False        
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
        self.num_attention_heads = num_attention_heads
        self.text_multihead = nn.MultiheadAttention(
            self.intermediate_dims, num_attention_heads
        )
        self.image_multihead = nn.MultiheadAttention(
            self.intermediate_dims, num_attention_heads
        )

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
        # Multi-Head Attention
        text_attended, _ = self.text_multihead(
            self.text_attention(encoded_text['last_hidden_state']),
            self.text_attention(encoded_text['last_hidden_state']),
            self.text_attention(encoded_text['last_hidden_state']),
        )
        image_attended, _ = self.image_multihead(
            self.image_attention(encoded_image['last_hidden_state']),
            self.image_attention(encoded_image['last_hidden_state']),
            self.image_attention(encoded_image['last_hidden_state']),
        )
        # Co-Attention between Image and Text
        text_attended = self.attention_weights(torch.tanh(text_attended))
        image_attended = self.attention_weights(torch.tanh(image_attended))
        attention_weights = torch.softmax(torch.cat([text_attended, image_attended], dim=1), dim=1)
        attended_text = torch.sum(attention_weights[:, 0].unsqueeze(-1) * encoded_text['last_hidden_state'], dim=1)
        attended_image = torch.sum(attention_weights[:, 1].unsqueeze(-1) * encoded_image['last_hidden_state'], dim=1)
        fused_output = self.fusion(torch.cat([attended_text, attended_image], dim=1))
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
        num_attention_heads=config["model"]['heads'],
        dropout=config["model"]["dropout"],
        pretrained_text_name=config["model"]["text_encoder"],
        pretrained_image_name=config["model"]["image_encoder"]
    )

    return model
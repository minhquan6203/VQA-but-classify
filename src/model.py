from typing import List, Dict, Optional
import torch
from torch import nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, pretrained_name):
        super(TextEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name, return_dict=True)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.encoder.config.hidden_size,
            num_heads=8,
            dropout=0.1
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        attention_output, _ = self.attention(sequence_output, sequence_output, sequence_output)
        pooled_output = self.encoder.pooler(attention_output)
        return pooled_output


class ImageEncoder(nn.Module):
    def __init__(self, pretrained_name):
        super(ImageEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name, return_dict=True)
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.encoder.config.hidden_size,
            num_heads=8,
            dropout=0.1
        )

    def forward(self, pixel_values):
        features = self.conv(pixel_values)
        features = features.view(features.size(0), -1)
        features = self.encoder.features(features)
        features = features.view(features.size(0), features.size(1), -1)
        features = features.permute(0, 2, 1)
        attention_output, _ = self.attention(features, features, features)
        pooled_output = self.encoder.pooler(attention_output)
        return pooled_output



class MultimodalVQAModel(nn.Module):
    def __init__(self, num_labels, intermediate_dims, dropout, pretrained_text_name, pretrained_image_name):
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.text_encoder = TextEncoder(pretrained_text_name)
        self.image_encoder = ImageEncoder(pretrained_image_name)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.text_encoder.encoder.config.hidden_size,
            num_heads=8,
            dropout=0.1
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.encoder.config.hidden_size + self.image_encoder.encoder.config.hidden_size, intermediate_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(intermediate_dims, self.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, pixel_values, attention_mask=None, token_type_ids=None, labels=None):
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        image_output = self.image_encoder(pixel_values=pixel_values)

        # Apply attention between the text and image outputs
        text_output = text_output.unsqueeze(0)
        image_output = image_output.unsqueeze(0)
        attention_output, _ = self.attention(text_output, image_output, image_output)
        text_output = text_output.squeeze(0)
        image_output = image_output.squeeze(0)
        attention_output = attention_output.squeeze(0)

        # Concatenate the text, image, and attention outputs
        fused_output = self.fusion(torch.cat([text_output, image_output, attention_output], dim=1))
        logits = self.classifier(fused_output)

        out = {"logits": logits}

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

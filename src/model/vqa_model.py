from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embedding
from vision_module.vision_embedding import  Vision_Encode_Feature,Vision_Embedding,VisionOcrObjEmbedding
from encoder_module.init_encoder import build_encoder
from data_utils.load_data import create_ans_space

class VQA_Model(nn.Module):
    def __init__(self,config: Dict):
     
        super(VQA_Model, self).__init__()
        self.num_labels = len(create_ans_space(config))
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.num_attention_heads=config["attention"]['heads']
        self.d_text = config["text_embedding"]['d_features']
        self.d_vision = config["vision_embedding"]['d_features']

        self.text_embedding = build_text_embedding(config)
        self.processor = Vision_Encode_Feature(config)
        self.vision_embedding=Vision_Embedding(config)
        self.use_ocr_obj=config['ocr_obj_embedding']['use_ocr_obj']
        if self.use_ocr_obj:
            self.ocr_obj_embedding=VisionOcrObjEmbedding(config)

        self.fusion = nn.Sequential(
            nn.Linear(self.intermediate_dims +self.intermediate_dims, self.intermediate_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.encoder = build_encoder(config)
        self.classifier = nn.Linear(self.intermediate_dims, self.num_labels)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, questions: List[str], images: List[str], labels: Optional[torch.LongTensor] = None):
        pixel_values, ocr_info, obj_info= self.processor(images)
        if ocr_info is not None:
            ocr_texts = [' '.join(t['texts']) if t['texts'] is not None else '' for t in ocr_info]
        embedded_text, text_mask = self.text_embedding(questions,ocr_texts)
        embedded_vision, vison_mask = self.vision_embedding(pixel_values)
        if self.use_ocr_obj:
            embedded_ocr_obj,ocr_obj_mask=self.ocr_obj_embedding(ocr_info, obj_info)
            embedded_vision=torch.cat([embedded_vision,embedded_ocr_obj],dim=1)
            vison_mask=torch.cat([vison_mask,ocr_obj_mask],dim=2)

        encoded_image, encoded_text = self.encoder(embedded_vision, vison_mask,embedded_text, text_mask)
        text_attended = self.attention_weights(torch.tanh(encoded_text))
        image_attended = self.attention_weights(torch.tanh(encoded_image))
        
        attention_weights = torch.softmax(torch.cat([text_attended, image_attended], dim=1), dim=1)
        
        attended_text = torch.sum(attention_weights[:, 0].unsqueeze(-1) * encoded_text, dim=1)
        attended_image = torch.sum(attention_weights[:, 1].unsqueeze(-1) * encoded_image, dim=1)
        
        fused_output = self.fusion(torch.cat([attended_text, attended_image], dim=1))
        logits = self.classifier(fused_output)
        logits = F.log_softmax(logits, dim=-1)

        if labels is not None:
            # logits=logits.view(-1,self.num_labels)
            # labels = labels.view(-1)
            loss = self.criterion(logits, labels)
            return logits,loss
        else:
            return logits

def createVQA_Model(config: Dict) -> VQA_Model:
    model = VQA_Model(config)
    return model
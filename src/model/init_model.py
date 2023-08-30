from model.vqa_model import VQA_Model
from typing import List, Dict, Optional

def build_model(config: Dict) -> VQA_Model:
    model = VQA_Model(config)
    return model


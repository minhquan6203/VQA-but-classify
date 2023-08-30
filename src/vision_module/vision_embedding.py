import torch
from torch import nn
import os
from PIL import Image
from typing import List
from transformers import AutoModel, AutoFeatureExtractor
from typing import List, Dict, Optional, Any
import numpy as np
from mask.masking import generate_padding_mask
class Vision_Encode_Feature(nn.Module):
    def __init__(self, config: Dict):
        super(Vision_Encode_Feature,self).__init__()
        self.preprocessor = AutoFeatureExtractor.from_pretrained(config["vision_embedding"]["image_encoder"])
        self.data_folder = config["data"]["data_folder"]
        self.image_folder = config["data"]["images_folder"]
        self.use_ocr_obj = config['ocr_obj_embedding']['use_ocr_obj']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.ocr_features_path = config['ocr_obj_embedding']['path_ocr']
        self.scene_text_threshold = config['ocr_obj_embedding']['threshold']
        self.max_scene_text = config['ocr_obj_embedding']['max_scene_text']
        self.d_det=config['ocr_obj_embedding']['d_det']
        self.d_rec=config['ocr_obj_embedding']['d_rec']
         
        self.obj_features_path = config['ocr_obj_embedding']['path_obj']
        self.max_bbox = config["ocr_obj_embedding"]['max_bbox']
        self.d_obj = config['ocr_obj_embedding']['d_obj']
        self.d_grid = config['ocr_obj_embedding']['d_grid']

    def forward(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[
                self.load_image(image_id) for image_id in images
            ],
            return_tensors="pt",
        ).to(self.device)

        if self.use_ocr_obj:
            ocr_info = [self.load_ocr_features(image_id) for image_id in images]
            obj_info =[self.load_obj_features(image_id) for image_id in images]
        else:
            ocr_info = None
            obj_info = None
        return processed_images.pixel_values, ocr_info, obj_info

    def load_image(self, image_id):
        image_path = os.path.join(self.data_folder, self.image_folder, str(image_id).zfill(11))
        #image_path = os.path.join(self.data_folder, self.image_folder, str(image_id))

        if os.path.exists(image_path + ".jpg"):
            image = Image.open(image_path + ".jpg").convert('RGB')
        elif os.path.exists(image_path + ".png"):
            image = Image.open(image_path + ".png").convert('RGB')
        elif os.path.exists(image_path + ".jpeg"):   
            image = Image.open(image_path + ".jpeg").convert('RGB')
        elif os.path.exists(image_path + ".JPG"):
            image = Image.open(image_path + ".JPG").convert('RGB')  
        else:
            raise FileNotFoundError(f"Image not found for {image_id}")

        return image
    
    def pad_array(self, array: np.ndarray, max_len: int, value):
        pad_value_array = np.zeros((max_len-array.shape[0], array.shape[-1])).fill(value)
        array = np.concatenate([array, pad_value_array], axis=0)
        
        return array

    def pad_tensor(self, tensor: torch.Tensor, max_len: int, value):
        pad_value_tensor = torch.zeros((max_len-tensor.shape[0], tensor.shape[-1])).fill_(value)
        tensor = torch.cat([tensor, pad_value_tensor], dim=0)
        
        return tensor

    def pad_list(self, list: List, max_len: int, value):
        pad_value_list = [value] * (max_len - len(list))
        list.extend(pad_value_list)

        return list
    
    def load_obj_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.obj_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        if os.path.exists(feature_file):
            for key, feature in features.items():
                if isinstance(feature, np.ndarray):
                    features[key] = torch.tensor(feature)
    
            if features['region_features'].shape[0] > self.max_bbox:
                region_features=features['region_features'][:self.max_bbox]
            else:
                region_features=self.pad_tensor(features['region_features'],self.max_bbox,1.)
            
            if features['region_boxes'].shape[0] > self.max_bbox:
                region_boxes=features['region_boxes'][:self.max_bbox]
            else:
                region_boxes=self.pad_tensor(features['region_boxes'],self.max_bbox,1.)
            
            obj_info={
                'region_features': region_features.detach().cpu(),
                'region_boxes': region_boxes.detach().cpu(),
                'grid_features': features['grid_features'].detach().cpu(),
                'grid_boxes': features['grid_boxes'].squeeze(0).detach.cpu(),
            }
        else:
            region_features=self.pad_tensor(torch.ones(1,self.d_obj), self.max_bbox, 1.)
            region_boxes=self.pad_tensor(torch.ones(1,4), self.max_bbox, 1.)
            grid_features=self.pad_tensor(torch.ones(1,self.d_grid), 49, 1.)
            grid_boxes=self.pad_tensor(torch.ones(1,4), 49, 1.)
            obj_info={
                'region_features': region_features.detach().cpu(),
                'region_boxes': region_boxes.detach().cpu(),
                'grid_features': grid_features.detach().cpu(),
                'grid_boxes': grid_boxes.detach().cpu(),
            }

        return obj_info
    
    def load_ocr_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.ocr_features_path, f"{image_id}.npy")
        if os.path.exists(feature_file):
            features = np.load(feature_file, allow_pickle=True)[()]
            for key, feature in features.items():
                if isinstance(feature, np.ndarray):
                    features[key] = torch.tensor(feature)

            # select ocr features and tokens having confident score greater than a threshold
            selected_ids = (np.array(features["scores"]) >= self.scene_text_threshold).tolist()
            for key, feature in features.items():
                if isinstance(feature, torch.Tensor) or isinstance(feature, np.ndarray):
                    feature = feature[selected_ids]
                else:
                    feature = [feature[idx] for idx, selected_id in enumerate(selected_ids) if selected_id]
                features[key] = feature
            # get the top confident-score ocr features and tokens
            if np.array(selected_ids).sum() > self.max_scene_text:
                topk_scores = torch.topk(torch.tensor(features["scores"]), k=self.max_scene_text)
                for key, feature in features.items():
                    if isinstance(feature, torch.Tensor):
                        feature = feature[topk_scores.indices]
                    else:
                        feature = [feature[idx] for idx in topk_scores.indices]
                    features[key] = feature
            else: # pad to the highest number of ocr tokens
                features['det_features'] = self.pad_tensor(features['det_features'], self.max_scene_text, 1.)
                features['rec_features'] = self.pad_tensor(features['rec_features'], self.max_scene_text, 1.)
                features['boxes'] = self.pad_tensor(features['boxes'], self.max_scene_text, 1.)
                # features['texts'] = self.pad_list(features['texts'], self.max_scene_text, '<pad>')
            ocr_info={
                    "det_features": features["det_features"].detach().cpu(),
                    "rec_features": features["rec_features"].detach().cpu(),
                    "texts": features["texts"],
                    "boxes": features["boxes"].detach().cpu(),
                    }
        else:
            det_features = self.pad_tensor(torch.ones(1,self.d_det), self.max_scene_text, 1.)
            rec_features = self.pad_tensor(torch.ones(1,self.d_rec), self.max_scene_text, 1.)
            boxes = self.pad_tensor(torch.ones(1,4), self.max_scene_text, 1.)
            # texts = self.pad_list(['<pad>'], self.max_scene_text, '<pad>')
            texts=None
            ocr_info={
                    "det_features": det_features.detach().cpu(),
                    "rec_features": rec_features.detach().cpu(),
                    "texts": texts,
                    "boxes": boxes.detach().cpu(),
                    }
    
        return ocr_info

class Vision_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super(Vision_Embedding,self).__init__()
        self.backbone = AutoModel.from_pretrained(config["vision_embedding"]["image_encoder"])
        # freeze all parameters of pretrained model
        if config["vision_embedding"]["freeze"]:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
        self.proj = nn.Linear(config["vision_embedding"]['d_features'], config["vision_embedding"]['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["vision_embedding"]['dropout'])
    def forward(self, pixel_values: List[str]):
        features = (self.backbone(pixel_values).last_hidden_state)
        padding_mask = generate_padding_mask(features, padding_idx=0)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask
    
class VisionOcrObjEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_det_features = nn.Linear(config['ocr_obj_embedding']['d_det'], config['ocr_obj_embedding']['d_model'])
        self.linear_rec_features = nn.Linear(config['ocr_obj_embedding']['d_rec'], config['ocr_obj_embedding']['d_model'])
        self.linear_bbox = nn.Linear(4, config['ocr_obj_embedding']['d_model'])

        self.linear_region_features = nn.Linear(config['ocr_obj_embedding']['d_obj'],config['ocr_obj_embedding']['d_model'])
        self.linear_region_boxes = nn.Linear(4,config['ocr_obj_embedding']['d_model'])
        self.linear_grid_features=nn.Linear(config['ocr_obj_embedding']['d_grid'], config['ocr_obj_embedding']['d_model'])
        self.linear_grid_bbox=nn.Linear(4, config['ocr_obj_embedding']['d_model'])
        
        self.layer_norm = nn.LayerNorm(config['ocr_obj_embedding']['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self,ocr_info,obj_info):
        det_features = torch.stack([det["det_features"] for det in ocr_info]).to(self.device)
        rec_features = torch.stack([rec["rec_features"] for rec in ocr_info]).to(self.device)
        boxes = torch.stack([bbox["boxes"] for bbox in ocr_info]).to(self.device)

        region_features=torch.stack([region["region_features"] for region in obj_info]).to(self.device)
        region_boxes=torch.stack([region["region_boxes"] for region in obj_info]).to(self.device)
        
        grid_features=torch.stack([grid["grid_features"] for grid in obj_info]).to(self.device)
        grid_boxes=torch.stack([grid["grid_boxes"] for grid in obj_info]).to(self.device)
        
        det_features=self.linear_det_features(det_features)
        rec_features=self.linear_det_features(rec_features)
        boxes=self.linear_bbox(boxes)

        region_features=self.linear_region_features(region_features)
        region_boxes =self.linear_region_boxes(region_boxes)
        grid_features=self.linear_grid_features(grid_features)
        grid_boxes=self.linear_grid_bbox(grid_boxes)

        ocr_features=torch.cat([det_features, rec_features, boxes],dim=1)
        obj_features=torch.cat([region_features, region_boxes,grid_features,grid_features],dim=1)
        ocr_obj_features = torch.cat([ocr_features,obj_features], dim=1)
        padding_mask = generate_padding_mask(ocr_obj_features, padding_idx=0)
        out = self.dropout(self.gelu(self.layer_norm(ocr_obj_features)))
        return out, padding_mask
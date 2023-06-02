import torch
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
import json
import os
from datasets import load_dataset

class Load_Data:
    def __init__(self, config: Dict):
        self.data_folder = config['data']['data_folder']
        self.num_worker = config['data']['num_worker']
    
    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        question = item["question"]
        answer = item["answer"]
        image_id = item['image_id']
        
        return{
            'question':question,
            'image_id':image_id,
            'answer':answer
        }
        
    def load_annotations(self, json_name) -> List[Dict]:
        with open(os.path.join(self.data_folder,json_name)) as f:
            json_data =json.load(f)
        annotations = []
        for ann in json_data["annotations"]:
            question = ann["question"]
            answer = ann['answers'][0]
            annotation = {
                "question": question,
                "answer": answer,
                "image_id": ann["image_id"],
            }
            annotations.append(annotation)
        return annotations
   
    def get_dataloader(self,json_name,batch_size):
        dataset=self.load_annotations(json_name)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_worker,
        )
        return dataloader



def create_ans_space(config: Dict):
    data_folder=config['data']['data_folder']
    train_set=config["data"]["train_dataset"]
    val_set=config["data"]["val_dataset"]
    test_set=config["data"]["test_dataset"]
    dataset = load_dataset(
        "json", 
        data_files={
            "train": os.path.join(data_folder, train_set),
            "val": os.path.join(data_folder, val_set),
            "test": os.path.join(data_folder, test_set)
        },field='annotations'
    )

    answer_space = []

    for data_file in dataset.values():
        for ans in data_file['answers']:
            answer_space.append(ans[0])
    answer_space = sorted(list(set(answer_space)))

    return answer_space
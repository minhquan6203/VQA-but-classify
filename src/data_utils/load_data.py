import torch
from typing import List, Dict, Optional
from torch.utils.data import DataLoader, Dataset
import json
import os
from datasets import load_dataset
from utils.utils import preprocess_sentence

class VQA_dataset(Dataset):
    def __init__(self, path, with_answer):
        with open(path, 'r') as file:
            json_data = json.load(file)
        self.annotations=self.load_annotations(json_data, with_answer)
    
    def load_annotations(self, json_data, with_answer=True) -> List[Dict]:
        annotations = []
        if with_answer:
            for ann in json_data["annotations"]:
                question = preprocess_sentence(ann["question"])
                answer = preprocess_sentence(ann['answers'][0])
                #answer = preprocess_sentence(ann['answer'])
                annotation = {
                    "id": ann['id'],
                    "image_id": ann['image_id'],
                    "question": question.replace('?',''),
                    "answer": answer,
                }
                annotations.append(annotation)
        else:
            for ann in json_data["annotations"]:
                question = preprocess_sentence(ann["question"])
                answer = preprocess_sentence(ann['answers'][0])
                #answer = preprocess_sentence(ann['answer'])
                annotation = {
                    "id": ann['id'],
                    "image_id": ann['image_id'],
                    "question": question.replace('?',''),
                }
                annotations.append(annotation)
        return annotations
    def __getitem__(self, index):
        item = self.annotations[index]
        return item
    def __len__(self) -> int:
        return len(self.annotations)

class Load_Data:
    def __init__(self, config: Dict):
        self.num_worker = config['data']['num_worker']

        self.train_path = config['data']['train_dataset']
        self.valid_path=config["data"]["val_dataset"]
        self.test_path=config['infer']['test_dataset']

        self.train_batch=config['train']['per_device_train_batch_size']
        self.valid_batch=config['train']['per_device_valid_batch_size']
        self.test_batch=config['infer']['per_device_eval_batch_size']
    def load_train_dev(self):
        train_set=VQA_dataset(self.train_path,True)
        val_set=VQA_dataset(self.valid_path,True)
        train_loader = DataLoader(train_set, batch_size=self.train_batch, num_workers=self.num_worker,shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.valid_batch, num_workers=self.num_worker,shuffle=True)
        return train_loader, val_loader

    def load_test(self,with_answer):
        test_set=VQA_dataset(self.test_path,with_answer)
        test_loader = DataLoader(test_set, batch_size=self.test_batch, num_workers=self.num_worker,shuffle=False)
        return test_loader


def create_ans_space(config: Dict):
    train_set=config["data"]["train_dataset"]
    val_set=config["data"]["val_dataset"]
    # test_set=config["data"]["test_dataset"]
    dataset = load_dataset(
        "json", 
        data_files={
            "train": train_set,
            "val": val_set,
            # "test": test_set
        },field='annotations'
    )

    answer_space = []

    for data_file in dataset.values():
        for ans in data_file['answers']:
            answer_space.append(ans[0])
    answer_space = sorted(list(set(answer_space)))

    return answer_space
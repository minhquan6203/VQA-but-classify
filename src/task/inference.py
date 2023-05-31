import argparse
import os
import yaml
import logging
from typing import Text, Dict, List
import pandas as pd
from torch.utils.data import DataLoader
import torch
import transformers
from utils.builder import get_model
from sklearn.metrics import accuracy_score, f1_score
import json
class Predict:
    def __init__(self,config: Dict,answer_space):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.answer_space =answer_space
        self.model_name =config["model"]["name"]
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], config["model"]["name"])
        self.checkpoint_model=os.path.join(self.checkpoint_path,min(os.listdir(self.checkpoint_path), key=lambda x: int(x.split('-')[1])),"pytorch_model.bin")
        self.test_path=config['inference']['test_dataset']
        self.bath_size=config['inference']['batch_size']
        self.model = get_model(config,len(self.answer_space))
    def predict_submission(self):
        transformers.logging.set_verbosity_error()
        logging.basicConfig(level=logging.INFO)
    

    # Load the model
        logging.info("Loading the {0} model...".format(self.model_name))
        logging.info("best checkpoint at path: {0}".format(self.checkpoint_model))
        self.model.load_state_dict(torch.load(self.checkpoint_model))
        self.model.to(self.device)

        # Obtain the prediction from the model
        logging.info("Obtaining predictions...")
        test_set =self.get_dataloader(self.test_path)
        y_preds=[]
        gts=[]
        self.model.eval()
        with torch.no_grad():
            for item in test_set:
                output = self.model(item['question'],item['image_id'])
                preds = output["logits"].argmax(axis=-1).cpu().numpy()
                answers = [self.answer_space[i] for i in preds]
                y_preds.extend(answers)
                gts.extend(item['answer'])
        print('accuracy on test:', accuracy_score(gts,y_preds))
        print('f1 macro on test:', f1_score(gts,y_preds,average='macro'))
        print('f1 weighted on test:', f1_score(gts,y_preds,average='weighted'))
        data = {'preds': y_preds,'gts': gts }
        df = pd.DataFrame(data)
        df.to_csv('./submission.csv', index=False)

    def load_annotations(self, json_file) -> List[Dict]:
        with open(os.path.join(self.data_folder,json_file)) as f:
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

    def get_dataloader(self,file_path):
        dataset=self.load_annotations(file_path)
        dataloader = DataLoader(
            dataset,
            batch_size=self.bath_size,
            num_workers=2,
        )
        return dataloader

import argparse
import os
import yaml
import logging
from typing import Text, Dict, List
import pandas as pd
from data_utils.load_data import create_ans_space
import torch
import transformers
from model.init_model import build_model
from eval_metric.evaluate import WuPalmerScoreCalculator
from data_utils.load_data import  Load_Data
from tqdm import tqdm
import json
import shutil
class Predict:
    def __init__(self,config: Dict):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.answer_space = create_ans_space(config)
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], "best_model.pth")
        self.test_path=config['data']['test_dataset']
        self.with_answer=config['infer']['with_answer']
        self.batch_size=config['inference']['batch_size']
        self.train_folder=config['data']['images_folder']
        self.test_folder=config['infer']['images_folder']
        self.model = build_model(config)
        self.dataloader = Load_Data(config)
        self.compute_score = WuPalmerScoreCalculator()
    def move_img(self):
        os.makedirs(self.train_folder,exist_ok=True)
        if self.train_folder != self.test_folder:
            for f in os.listdir(self.test_folder):
                shutil.move(os.path.join(self.test_folder,f), self.train_folder)

    def predict_submission(self):
        transformers.logging.set_verbosity_error()
        logging.basicConfig(level=logging.INFO)
        self.move_img()
    # Load the model
        logging.info("loadding best model...")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Obtain the prediction from the model
        logging.info("Obtaining predictions...")
        test_set =self.dataloader.load_test()
        if self.with_answer:
            y_preds=[]
            gts=[]
            ids=[]
            self.model.eval()
            with torch.no_grad():
                for it, item in enumerate(tqdm(test_set)):
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        output = self.model(item['question'],item['image_id'].tolist())
                        preds = output.argmax(axis=-1).cpu().numpy()
                        answers = [self.answer_space[i] for i in preds]
                        y_preds.extend(answers)
                        gts.extend(item['answer'])
                        ids.extend(item['id'])
            print('accuracy on test:', self.compute_score.accuracy(gts,y_preds))
            print('f1 char on test:', self.compute_score.F1_char(gts,y_preds))
            print('f1 token on test:', self.compute_score.F1_token(gts,y_preds))
            print('wups on test:', self.compute_score.batch_wup_measure(gts,y_preds))
            data = {'id':ids, 'preds': y_preds,'gts': gts }
            df = pd.DataFrame(data)
            df.to_csv('./submission.csv', index=False)
        else:
            self.model.eval()
            y_preds={}
            with torch.no_grad():
                for it, item in enumerate(tqdm(test_set)):
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        output = self.model(item['question'],item['image_id'].tolist())
                        preds = output.argmax(axis=-1).cpu().numpy()
                        answers = [self.answer_space[i] for i in preds]
                        for i in range(len(answers)):
                            y_preds[str(item['id'][i])] = answers[i]
            with open('/content/results.json', 'w', encoding='utf-8') as r:
                json.dump(y_preds, r, ensure_ascii=False, indent=4)



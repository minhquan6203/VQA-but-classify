import argparse
import os
import yaml
import logging
from typing import Text, Dict, List
import pandas as pd
from data_utils.load_data import create_ans_space
import torch
import transformers
from utils.builder import get_model
from eval_metric.evaluate import WuPalmerScoreCalculator
from data_utils.load_data import  Load_Data
class Predict:
    def __init__(self,config: Dict):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.answer_space = create_ans_space(config)
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], "best_model.pth")
        self.test_path=config['data']['test_dataset']
        self.batch_size=config['inference']['batch_size']
        self.model = get_model(config)
        self.dataloader = Load_Data(config)
        self.compute_score = WuPalmerScoreCalculator()

    def predict_submission(self):
        transformers.logging.set_verbosity_error()
        logging.basicConfig(level=logging.INFO)
    
    # Load the model
        logging.info("loadding best model...")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Obtain the prediction from the model
        logging.info("Obtaining predictions...")
        test_set =self.dataloader.get_dataloader(self.test_path,self.batch_size)
        y_preds=[]
        gts=[]
        self.model.eval()
        with torch.no_grad():
            for item in test_set:
                output = self.model(item['question'],item['image_id'].tolist())
                preds = output["logits"].argmax(axis=-1).cpu().numpy()
                answers = [self.answer_space[i] for i in preds]
                y_preds.extend(answers)
                gts.extend(item['answer'])
        print('accuracy on test:', self.compute_score.accuracy(gts,y_preds))
        print('f1 char on test:', self.compute_score.F1_char(gts,y_preds))
        print('f1 token on test:', self.compute_score.F1_token(gts,y_preds))
        print('wups on test:', self.compute_score.batch_wup_measure(gts,y_preds))
        data = {'preds': y_preds,'gts': gts }
        df = pd.DataFrame(data)
        df.to_csv('./submission.csv', index=False)



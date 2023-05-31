import os
import yaml
import argparse
import logging
import json
from typing import Text
import torch
import transformers

from data_utils.load_data import loadDataset
from data_utils.data_collator import createDataCollator
from task.train import train_model
from eval_metric.evaluate import ScoreCalculator
from task.inference import Predict
from utils.builder import build_model

def training(config,device,data): 
    collator = createDataCollator(config)
    logging.info("Created data collator")
    
    model = build_model(config, data["answer_space"]).to(device)
    logging.info("Initialized model for training")
    
    calculator = ScoreCalculator(data["answer_space"])
    
    logging.info("Training started...")
    training_metrics, eval_metrics = train_model(
        config, device, data["dataset"], 
        collator, model,
        calculator.compute_metrics
    )
    
    logging.info("Training complete")
    
    os.makedirs(config["metrics"]["metrics_folder"], exist_ok=True)
    
    metrics = {**training_metrics[2], **eval_metrics}
    
    metrics_path = os.path.join(config["metrics"]["metrics_folder"], config["metrics"]["metrics_file"])
    json.dump(
        obj=metrics,
        fp=open(metrics_path, 'w'),
        indent=4
    )
    logging.info("Metrics saved")
 

def predicting(config,answer_space):
    logging.info("predicting...")
    predict=Predict(config,answer_space)
    predict.predict_submission()
    logging.info("task done!!!")

def main(config_path: Text) -> None:
    transformers.logging.set_verbosity_error()
    logging.basicConfig(level=logging.INFO)
    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    if config["base"]["use_cuda"]:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' # SET ONLY 1 GPU DEVICE
    else:
        device =  torch.device('cpu')
    data = loadDataset(config)
    logging.info("Loaded processed sentiment Dataset")
    
    training(config,device,data) #predict thì cmt lại, vào sửa config tới best model

    predicting(config,data["answer_space"])


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    main(args.config)
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
from transformers import  AutoModel, AutoTokenizer

from data_utils.load_data import Load_Data
from utils.builder import build_model
from eval_metric.evaluate import WuPalmerScoreCalculator

class STVQA_Task:
    def __init__(self, config):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.data_folder=config['data']['dataset_folder']
        self.train_path = config['data']['train_dataset']
        self.valid_path=config["data"]["val_dataset"]
        self.test_path=config["data"]["test_dataset"]
        self.learning_rate = config['train']['learning_rate']
        self.train_batch=config['train']['per_device_train_batch_size']
        self.test_batch=config['train']['per_device_eval_batch_size']
        self.valid_batch=config['train']['per_device_valid_batch_size']
        self.save_path = config['train']['output_dir']
        self.dataloader = Load_Data(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model=build_model(config).to(self.device)
        self.compute_score = WuPalmerScoreCalculator(config)
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate)

    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
    
        train = self.dataloader.get_dataloader(self.train_path,self.train_batch)
        valid = self.dataloader.get_dataloader(self.valid_path,self.valid_batch)

        
        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")
            train_loss = 0.
            valid_loss = 0.

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_valid_f1 = checkpoint['valid_f1']
        else:
            best_valid_f1 = 0.
            
        threshold=0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            valid_acc = 0.
            valid_wups=0.
            valid_f1 =0.
            train_loss = 0.
            valid_loss = 0.
            for item in train:
                self.optimizer.zero_grad()
                logits, loss = self.base_model(item['question'],item['image_id'],item['answer'])
                loss.backward()
                self.optimizer.step()
                train_loss += loss
                print(loss)
            train_loss /=len(train)
            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f}")
            
            with torch.no_grad():
                for item in valid:
                    self.optimizer.zero_grad()
                    logits, loss = self.base_model(item['question'],item['image_id'],item['answer'])
                    valid_loss += loss
                    wups,acc,f1 = self.compute_score.compute_metrics(item['answer'],logits)
                    valid_wups+=wups
                    valid_acc+=acc
                    valid_f1+=f1
            valid_loss /=len(valid)
            valid_wups /= len(valid)
            valid_acc /= len(valid)
            valid_f1 /= len(valid)
            
            print(f"valid loss: {valid_loss:.4f} valid wups: {valid_wups:.4f} valid acc: {valid_acc:.4f} valid f1: {valid_f1:.4f}")

            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'valid_f1': valid_f1}, os.path.join(self.save_path, 'last_model.pth'))

            # save the best model

            if epoch > 0 and valid_f1 < best_valid_f1:
              threshold += 1
            else:
              threshold = 0

            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'valid_f1': valid_f1,}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with validation f1 token level of {valid_f1:.4f}")
            
            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break


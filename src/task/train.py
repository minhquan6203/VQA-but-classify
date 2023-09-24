import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
from data_utils.load_data import Load_Data
from model.init_model import build_model
from eval_metric.evaluate import WuPalmerScoreCalculator
from data_utils.load_data import create_ans_space
from tqdm import tqdm

class STVQA_Task:
    def __init__(self, config):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.learning_rate = config['train']['learning_rate']
        self.save_path = config['train']['output_dir']
        self.best_metric= config['train']['metric_for_best_model']
        self.answer_space=create_ans_space(config)
        self.dataloader = Load_Data(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model=build_model(config).to(self.device)
        self.compute_score = WuPalmerScoreCalculator()
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler()

    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
    
        train,valid = self.dataloader.load_train_dev()
        
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
            best_score = checkpoint['score']
        else:
            best_score = 0.
            
        threshold=0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            valid_acc = 0.
            valid_wups=0.
            valid_f1 =0.
            train_loss = 0.
            for it, item in enumerate(tqdm(train)):
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    labels=torch.tensor([self.answer_space.index(answer) for answer in item['answer']]).to(self.device)
                    logits, loss = self.base_model(item['question'],item['image_id'].tolist(),labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                train_loss += loss
            train_loss /=len(train)
            
            with torch.no_grad():
                for it, item in enumerate(tqdm(valid)):
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        logits = self.base_model(item['question'],item['image_id'].tolist())
                    preds = logits.argmax(axis=-1).cpu().numpy()
                    answers = [self.answer_space[i] for i in preds]
                    valid_wups+=self.compute_score.batch_wup_measure(item['answer'],answers)
                    valid_acc+=self.compute_score.accuracy(item['answer'],answers)
                    valid_f1+=self.compute_score.F1_token(item['answer'],answers)
                    
            valid_wups /= len(valid)
            valid_acc /= len(valid)
            valid_f1 /= len(valid)

            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f}")
            print(f"valid wups: {valid_wups:.4f} valid acc: {valid_acc:.4f} valid f1: {valid_f1:.4f}")

            if self.best_metric =='accuracy':
                score=valid_acc
            if self.best_metric=='f1':
                score=valid_f1
            if self.best_metric=='wups':
                score=valid_wups

            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'score': score}, os.path.join(self.save_path, 'last_model.pth'))
            
            # save the best model
            if epoch > 0 and score < best_score:
              threshold += 1
            else:
              threshold = 0

            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'score':score}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with {self.best_metric} of {score:.4f}")
            
            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break


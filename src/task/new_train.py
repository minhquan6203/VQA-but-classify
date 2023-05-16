import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
from transformers import  AutoModel, AutoTokenizer

from data_utils.load_data_new import Load_Data
from model.model_gen import createMultimodalModelForVQA
from data_utils.load_data_new import create_ans_space
from decoder_module.decoder import Decoder
from eval_metric.evaluate import WuPalmerScoreCalculator
class STVQA_Task:
    def __init__(self, config):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.data_folder=config['data']['data_folder']
        self.train_path = config['data']['train_path']
        self.valid_path=config["data"]["val_dataset"]
        self.test_path=config["data"]["test_dataset"]
        self.learning_rate = config['train']['learning_rate']
        self.train_batch=config['train'][' per_device_train_batch_size']
        self.test_batch=config['train'][' per_device_train_test_size']
        self.valid_batch=config['train']['per_device_train_valid_size']
        self.save_path = config['train']['output_dir']
        self.dataloader = Load_Data(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.answer_space = create_ans_space(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        self.decoder = Decoder(config)
        self.base_model=createMultimodalModelForVQA(config).to(self.device)
        self.compute_score = WuPalmerScoreCalculator(config)

    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
    
        train = self.dataloader.get_dataloader(self.train_path,self.train_batch)
        valid = self.dataloader.get_dataloader(self.valid_path,self.valid_batch)

        optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate)
        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            best_valid_acc = checkpoint['valid_acc']
        else:
            best_valid_acc = 0.
            
        threshold=0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            valid_acc = 0.
            valid_wups=0.
            valid_f1 =0.
            train_loss = 0.
            valid_loss = 0.
            for item in train:
                optimizer.zero_grad()
                fused_output, fused_mask  = self.base_model(item['question'],item['image_id'],item['answer'])
                labels = self.tokenizer.batch_encode_plus(item['answer'],padding='max_length',truncation=True,max_length=fused_output.shape[1],return_tensors='pt').to(self.device)
                logits, loss = self.decoder(fused_output,fused_mask,labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            with torch.no_grad():
                for item in valid:
                    optimizer.zero_grad()
                    fused_output, fused_mask  = self.base_model(item['question'],item['image_id'],item['answer'])
                    labels = self.tokenizer.batch_encode_plus(item['answer'],padding='max_length',truncation=True,max_length=fused_output.shape[1],return_tensors='pt').to(self.device)
                    logits, loss = self.decoder(fused_output,fused_mask,labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    wups,acc,f1 = self.compute_score(logits,item['answer'])
                    valid_wups+=wups
                    valid_acc+=acc
                    valid_f1+=f1
            train_loss /= len(train)
            valid_loss /= len(valid)
            valid_wups /= len(valid)
            valid_acc /= len(valid)
            valid_f1 /= len(valid)
            

            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f}")
            print(f"valid loss: {valid_loss:.4f} valid wups: {valid_wups:.4f} valid acc: {valid_acc:.4f} valid f1: {valid_f1}")

            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_acc': valid_acc}, os.path.join(self.save_path, 'last_model.pth'))

            # save the best model

            if epoch > 0 and valid_acc < best_valid_acc:
              threshold += 1
            else:
              threshold = 0

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_acc': valid_acc,}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with validation accuracy of {valid_acc:.4f}")
            
            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break


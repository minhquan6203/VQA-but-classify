import json
import pandas as pd
import pathlib
from numpy.random import RandomState
import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

class Data_Preroccessing:
    def __init__(self, config: Dict):
      self.json_file_train= config['old_data']['train_dataset']
      self.json_file_dev= config['old_data']['val_dataset']
      self.json_file_test= config['old_data']['test_dataset']

      self.new_folder = config['data']['dataset_folder']
      self.answer_space = config['data']['answer_space']
      self.all_qa_pairs = config['data']['all_qa_pairs_file']
      self.images_folder = config['old_data']['images_folder']
      self.resize = config['old_data']['resize']

    def preprocess(self, json_file, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        # Read data
        f = open(json_file,encoding='utf-8')
        data = json.load(f)
        
        #convert to csv file
        df_qa = pd.DataFrame.from_dict(data['annotations'])

        for i in range(len(df_qa['image_id'])):
            df_qa['image_id'][i]=str(df_qa['image_id'][i])
            df_qa['question'][i]=str(df_qa['question'][i])
            if len(df_qa['answers'][i][0])!=0 and df_qa['answers'][i][0] != 'nan':
              df_qa['answers'][i]=str(df_qa['answers'][i][0])
            else:
              df_qa['answers'][i]='NULL'
    
        # answer_space.txt
        with open(os.path.join(output_folder,'answer_space.txt'), 'a',encoding='utf-8') as f:
            for i in range(len(df_qa['answers'])):
              if len(df_qa['answers'][i][0]) !=0 and df_qa['answers'][i][0] != 'nan':
                  f.write(df_qa['answers'][i])
              else:
                f.write('NULL')
              f.write('\n')

        #all_qa_pairs.txt
        with open(os.path.join(output_folder,'all_qa_pairs.txt'), 'a',encoding='utf-8') as f:
            for i in range(len(df_qa['answers'])):
              f.write(df_qa['question'][i])
              f.write('\n')
              if len(df_qa['answers'][i][0]) !=0 and df_qa['answers'][i][0] != 'nan':
                  f.write(df_qa['answers'][i])
              else:
                f.write('đéo biết')
              f.write('\n')
        df_qa.to_csv(os.path.join(output_folder,json_file.replace('json','csv')))

    def move_images(self,input_folder, output_folder):
      os.makedirs(output_folder,exist_ok=True)
      for filename in os.listdir(input_folder):
          shutil.move(os.path.join(input_folder,filename),output_folder)

    def resize_images(self, input_folder, output_folder, size):
      img_error=[]

      os.makedirs(output_folder,exist_ok=True)
      for filename in os.listdir(input_folder):
        try:
          if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
              with Image.open(os.path.join(input_folder, filename)) as im:
                  if im.mode != "RGB":
                      im = im.convert("RGB")
                  im_resized = im.resize(size)
                  output_filename = os.path.join(output_folder, filename)
                  im_resized.save(output_filename)
        except:
          img_error.append(filename)
      print(img_error)


    def convert(self):
      if self.resize:
          size=(self.image_H,self.image_H)
          self.resize_images(self.images_folder,os.path.join(self.new_folder,'images'),size)
      else:
          self.move_images(self.images_folder,os.path.join(self.new_folder,'images'))
      
      if os.path.exists(os.path.join(self.new_folder,'answer_space.txt')):
          os.remove(os.path.join(self.new_folder,'answer_space.txt'))

      if os.path.exists(os.path.join(self.new_folder,'all_qa_pairs.txt')):
          os.remove(os.path.join(self.new_folder,'all_qa_pairs.txt'))
      self.preprocess(self.json_file_train, self.new_folder)
      self.preprocess(self.json_file_dev, self.new_folder)
      self.preprocess(self.json_file_test, self.new_folder)

       
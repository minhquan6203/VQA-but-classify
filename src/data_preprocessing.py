import json
import pandas as pd
import pathlib
from numpy.random import RandomState
import os
from PIL import Image
from sklearn.model_selection import train_test_split 

def preprocess(json_file,folder_output,img_error,path_img):
    os.makedirs(folder_output, exist_ok=True)
    path=folder_output
    # Read data
    f = open(json_file,encoding='utf-8')
    data = json.load(f)
    
    #convert to csv file
    df_qa = pd.DataFrame.from_dict(data['annotations'])

    for i in range(len(df_qa['image_id'])):
        df_qa['image_id'][i]=str(df_qa['image_id'][i])
        df_qa['question'][i]=str(df_qa['question'][i])
        if len(df_qa['answer'][i])!=0 and df_qa['answer'][i] != 'nan':
          df_qa['answer'][i]=str(df_qa['answer'][i])
        else:
          df_qa['answer'][i]='đéo biết'
    

    for i in range(len(img_error)):
      img_error[i]=img_error[i].replace('.jpg','')
    print(img_error)
    df_qa = df_qa[~df_qa['image_id'].isin(img_error)]

    file_img=os.listdir(path_img)
    for i in range(len(file_img)):
      file_img[i]=file_img[i].replace('.jpg','')
    df_qa = df_qa[df_qa['image_id'].isin(file_img)]

    df_qa.to_csv(f'{path}/data1.csv',index=False)
    df_qa=pd.read_csv(f'{path}/data1.csv')

    # answer_space.txt
    p = pathlib.Path(f'{path}/answer_space.txt')
    p.touch()
    with open(f'{path}/answer_space.txt', 'w',encoding='utf-8') as f:
        for i in range(len(df_qa['answer'])):
          if len(df_qa['answer'][i]) !=0 and df_qa['answer'][i] != 'nan':
              f.write(df_qa['answer'][i])
          else:
            f.write('đéo biết')
          f.write('\n')

    with open(f'{path}/all_qa_pairs.txt', 'w',encoding='utf-8') as f:
        for i in range(len(df_qa['answer'])):
          f.write(df_qa['question'][i])
          f.write('\n')
          if len(df_qa['answer'][i]) !=0 and df_qa['answer'][i] != 'nan':
              f.write(df_qa['answer'][i])
          else:
            f.write('đéo biết')
          f.write('\n')

    #split file for train, validation and test
    df = pd.read_csv(f'{path}/data1.csv')
    rng = RandomState()
    df = pd.read_csv(f'{path}/data1.csv')
    rng = RandomState()
    train_val, test = train_test_split(df, test_size=0.1, random_state=rng)
    train, val = train_test_split(train_val, test_size=0.1111, random_state=rng)


    train.to_csv(f'{path}/train.csv',index=False)
    val.to_csv(f'{path}/val.csv',index=False)
    test.to_csv(f'{path}/test.csv',index=False)

def resize_images(input_folder, output_folder, size):
    img_error=[]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
    
    return img_error

if __name__ == '__main__':

    img_error=resize_images('./book_fahasa','./data_fahasa/book_fahasa',(512,512))
    print(img_error)

    json_file = '/content/drive/MyDrive/vivqa_on_book/new_b_all_fahasa_label.json'
    folder_output = 'data_fahasa'
    preprocess(json_file, folder_output,img_error,'./data_fahasa/book_fahasa')
import json
import pandas as pd
import pathlib
from numpy.random import RandomState
import os
from PIL import Image

def preprocess(json_file,folder_output):
    os.makedirs(folder_output, exist_ok=True)
    path=folder_output
    # Read data
    f = open(json_file,encoding='utf-8')
    data = json.load(f)

    #convert to csv file
    df_image= pd.DataFrame.from_dict(data['images'])
    df_qa = pd.DataFrame.from_dict(data['annotations'])
    for i in range(len(df_qa['image_id'])):
        df_qa['image_id'][i]=str(df_qa['image_id'][i])
        df_qa['question'][i]=str(df_qa['question'][i])
        if len(df_qa['answer'][i])==0:
          df_qa['answer'][i]='đéo biết'
    df_qa1=df_qa
    df_qa1.to_csv(f'{path}/data1.csv',index=False)

    # answer_space.txt
    p = pathlib.Path(f'{path}/answer_space.txt')
    p.touch()
    with open(f'{path}/answer_space.txt', 'w',encoding='utf-8') as f:
        for i in range(len(df_qa['answer'])):
          if len(df_qa['answer'][i]) !=0:
              f.write(df_qa['answer'][i])
          else:
            f.write('đéo biết')
          f.write('\n')

    with open(f'{path}/all_qa_pairs.txt', 'w',encoding='utf-8') as f:
        for i in range(len(df_qa['answer'])):
          f.write(df_qa['question'][i])
          f.write('\n')
          if len(df_qa['answer'][i]) !=0:
              f.write(df_qa['answer'][i])
          else:
            f.write('đéo biết')
          f.write('\n')

    #split file for train, validation and test
    df = pd.read_csv(f'{path}/data1.csv')
    rng = RandomState()
    train_val = df.sample(frac=0.8, random_state=rng)
    test = df.loc[~df.index.isin(train_val.index)]
    train = train_val.sample(frac=0.8889, random_state=rng)
    val = train_val.loc[~train_val.index.isin(train.index)]

    train.to_csv(f'{path}/train.csv',index=False)
    val.to_csv(f'{path}/val.csv',index=False)
    test.to_csv(f'{path}/test.csv',index=False)

def resize_images(input_folder, output_folder, size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(os.path.join(input_folder, filename)) as im:
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im_resized = im.resize(size)
                output_filename = os.path.join(output_folder, filename)
                im_resized.save(output_filename)


if __name__ == '__main__':
    json_file = '/content/drive/MyDrive/vivqa_on_book/new_b_all_label.json'
    folder_output = 'data'
    preprocess(json_file, folder_output)
    resize_images('book_cover_img','./data/book_cover_img',(512,512))

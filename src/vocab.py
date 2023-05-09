import json
from collections import Counter
import os
def make_vocab(config):
    question_vocab = Counter()
    answer_vocab = Counter()
    list_data=[config.data.train_dataset,config.data.val_dataset,config.data.test_dataset]
    for split in list_data:
        
        with open(os.join.path(config.data.dataset_folder,split), 'r') as f:
            data = json.load(f)

        # Extract answers and update answer vocabulary
        answers = [ans.lower() for ann in data['annotations'] for ans in ann['answers']]
        answer_vocab.update(answers)

        # Extract questions and update question vocabulary
        questions = [ann['question'].lower() for ann in data['annotations']]
        question_vocab.update(questions)

    # Write answer space to file
    with open('answer_space.txt', 'w') as f:
        for answer in answer_vocab:
            f.write(answer + '\n')

    # Write question and answer vocabularies to file
    with open('vocab.txt', 'w') as f:
        for word, count in question_vocab.most_common():
            f.write(f'{word} {count}\n')
        for word, count in answer_vocab.most_common():
            f.write(f'{word} {count}\n')

from typing import List, Dict, Optional
from datasets import load_dataset
import os

def create_vocab(config: Dict):
    dataset = load_dataset(
        "json", 
        data_files={
            "train": os.path.join(config["data"]["dataset_folder"], config["data"]["train_dataset"]),
            "val": os.path.join(config["data"]["dataset_folder"], config["data"]["val_dataset"]),
            "test": os.path.join(config["data"]["dataset_folder"], config["data"]["test_dataset"])
        },field='annotations'
    )

    word_counts = {}

    for data_file in dataset.values():
        try:
            for ques in data_file['question']:
                for word in ques.split():
                    word=word.lower()
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
                        
            for ans in data_file['answers']:
                for word in ans[0].split():
                    word=word.lower()
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
        except:
            pass
    word_counts['unknown'] = len(word_counts) + 1
    sorted_word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
    vocab = list(sorted_word_counts.keys())

    return vocab, sorted_word_counts
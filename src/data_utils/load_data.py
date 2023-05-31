import os
from typing import Dict
from datasets import load_dataset

def loadDataset(config: Dict) -> Dict:
    dataset = load_dataset(
        "json", 
        data_files={
            "train": os.path.join(config["data"]["dataset_folder"], config["data"]["train_dataset"]),
            "val": os.path.join(config["data"]["dataset_folder"], config["data"]["val_dataset"]),
            "test": os.path.join(config["data"]["dataset_folder"], config["data"]["test_dataset"])
        },field='annotations'
    )

    answer_space = []

    for data_file in dataset.values():
        for ans in data_file['answers']:
            answer_space.append(ans[0])
    answer_space = list(set(answer_space))  # Remove duplicates and convert to list
    # output_file = os.path.join(config["data"]["dataset_folder"], config["data"]["answer_space"])

    # with open(output_file, "w") as f:
    #     for answer in answer_space:
    #         f.write(answer + "\n")
    
    dataset = dataset.map(
    lambda examples: {
        'label': [
            answer_space.index(ans[0]) # Select the 1st answer if multiple answers are provided
            for ans in examples['answers']]
    },
    batched=True)


    return {
        "dataset": dataset,
        "answer_space": answer_space
    }

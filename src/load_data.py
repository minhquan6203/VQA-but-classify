import os
from typing import Dict
from datasets import load_dataset

def loadDaquarDataset(config: Dict) -> Dict:
    dataset = load_dataset(
        "json", 
        data_files={
            "train": os.path.join(config["data"]["dataset_folder"], config["data"]["train_dataset"]),
            "val": os.path.join(config["data"]["dataset_folder"], config["data"]["val_dataset"]),
            "test": os.path.join(config["data"]["dataset_folder"], config["data"]["test_dataset"])
        }
    )

    answer_space = []

    for data_file in dataset.values():
        for ann in data_file:
            for ans in ann['annotations']:
                answer = ans['answers'][0]
                answer_space.append(answer)
    answer_space = list(set(answer_space))  # Remove duplicates and convert to list

    # output_file = os.path.join(config["data"]["dataset_folder"], config["data"]["answer_space"])

    # with open(output_file, "w") as f:
    #     for answer in answer_space:
    #         f.write(answer + "\n")

    # with open(os.path.join(config["data"]["dataset_folder"], config["data"]["answer_space"])) as f:
    #     answer_space = f.read().splitlines()

    dataset = dataset.map(
        lambda examples: {'annotations': [ann for ann in examples['annotations']]},
        batched=True,
        batch_size=64,
    )
    return {
        "dataset": dataset,
        "answer_space": answer_space
    }

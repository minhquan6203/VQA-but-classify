import torch
from instance import Instance
from PIL import Image
from typing import Dict, List
import torch
from torch.utils import data
from process_text import preprocess_sentence
import json
import os
import numpy as np
from typing import Dict, List, Any

class BaseDataset(data.Dataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(BaseDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = vocab

        # quesion-answer pairs
        self.annotations = self.load_annotations(json_data)

        # image features
        self.image_path = config.DATASET.IMAGE_PATH

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        raise NotImplementedError

    def load_images(self, filename: int) -> Dict[str, Any]:
        image_file = os.path.join(self.image_path,filename)
        images=Image.open(image_file).convert('RGB')
        return images

    def __getitem__(self, idx: int):
        raise NotImplementedError("Please inherit the BaseDataset class and implement the __getitem__ method")

    def __len__(self) -> int:
        return len(self.annotations)

class RawQuestionImageDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    @property
    def questions(self):
        return [ann["question"] for ann in self.annotations]

    @property
    def answers(self):
        return [ann["answer"] for ann in self.annotations]

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    for answer in ann["answers"]:
                        question = ann["question"]
                        answer=preprocess_sentence(answer, self.vocab.tokenizer)
                        annotation = {
                            "question": question,
                            "answer": answer,
                            "image_id": ann["image_id"],
                            "filename": image["filename"]
                        }
                        annotations.append(annotation)
                    break

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        question = item["question"]
        answer = self.vocab.encode_answer(item["answer"])

        shifted_right_answer = torch.zeros_like(answer).fill_(self.vocab.padding_idx)
        shifted_right_answer[:-1] = answer[1:]
        answer = torch.where(answer == self.vocab.eos_idx, self.vocab.padding_idx, answer) # remove eos_token in answer
        images= self.load_images(self.annotations[idx]["filename"])
    

        return Instance(
            question=question,
            answer_tokens=answer,
            shifted_right_answer_tokens=shifted_right_answer,
            images=images
        )

    def __len__(self) -> int:
        return len(self.annotations)


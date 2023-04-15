import csv
from collections import Counter

def create_vocab(csv_file):
    question_vocab = Counter()
    answer_vocab = Counter()
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row['question'].lower().strip()
            answer = row['answer'].lower().strip()
            question_vocab.update(question.split())
            answer_vocab.update(answer.split())

    question_vocab = {word: i+1 for i, (word, count) in enumerate(question_vocab.items())}
    answer_vocab = {word: i+1 for i, (word, count) in enumerate(answer_vocab.items())}

    return question_vocab, answer_vocab

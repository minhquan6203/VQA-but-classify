import string
def countTrainableParameters(model) -> int:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def normalize_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return text
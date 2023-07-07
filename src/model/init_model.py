from model.vqa_model import createVQA_Model,VQA_Model


def build_model(config):
    return createVQA_Model(config)

def get_model(config):
    return VQA_Model(config)


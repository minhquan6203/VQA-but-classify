from model.mcan_model import createMCAN_Model,MCAN_Model
from model.vilbert__model import createViLBert_Model,ViLBert_Model

def build_model(config, answer_space):
    if config['model']['type_model']=='mcan':
        return createMCAN_Model(config, answer_space)
    if config['model']['type_model']=='vilbert':
        return createViLBert_Model(config, answer_space)

def get_model(config, num_labels):
    if config['model']['type_model']=='mcan':
        return MCAN_Model(config, num_labels)
    if config['model']['type_model']=='vilbert':
        return ViLBert_Model(config, num_labels)

from model.mcan_model import createMCAN_Model,MCAN_Model
from model.vilbert_model import createViLBert_Model,ViLBert_Model

def build_model(config):
    if config['model']['type_model']=='mcan':
        return createMCAN_Model(config)
    if config['model']['type_model']=='vilbert':
        return createViLBert_Model(config)

def get_model(config):
    if config['model']['type_model']=='mcan':
        return MCAN_Model(config)
    if config['model']['type_model']=='vilbert':
        return ViLBert_Model(config)

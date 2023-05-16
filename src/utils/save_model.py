import torch
import torch.nn as nn

class Saving:
    def __init__(self,config):
        self.save_path = config['train']['output_dir']
def save_model(epoch,base_model,name):
    torch.save({
    'epoch': epoch,
    'model_state_dict': self.base_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'valid_acc': valid_acc,
    'train_acc': train_acc,
    'train_loss': train_loss,
    'valid_loss': valid_loss}, os.path.join(self.save_path, 'last_model.pth'))
    }
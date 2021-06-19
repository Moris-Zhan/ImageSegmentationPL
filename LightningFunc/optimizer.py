import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def configure_optimizers(self, mode = 'sgd'):
    if mode == 'sgd': 
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0, weight_decay=1e-5)
    elif mode == 'adam': 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
    return {
        'optimizer': optimizer,
        'lr_scheduler': ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9),
        'monitor': 'val_loss'
    }

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']    
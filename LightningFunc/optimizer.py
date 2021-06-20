import torch
from torch.optim import Adam, Adagrad, RMSprop, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LambdaLR, CyclicLR


def configure_optimizers(self):
    if self.args.optimizer == "Adam":
        optimizer = Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    elif self.args.optimizer == "Adagrad":
        optimizer = Adagrad(self.parameters(), lr=self.args.lr, lr_decay=self.args.lr_decay, weight_decay=self.args.weight_decay)
    elif self.args.optimizer == "RMSprop":
        optimizer = RMSprop(self.parameters(), lr=self.args.lr, alpha=self.args.alpha, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
    elif self.args.optimizer == "SGD": # default
        optimizer = SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

    if self.args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)
    elif self.args.lr_scheduler == "StepLR":
        lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.8)
    elif self.args.lr_scheduler == "MultiStepLR":
        lr_scheduler = MultiStepLR(optimizer, milestones=[70, 140, 190], gamma=0.1)
    elif self.args.lr_scheduler == "ExponentialLR":
        lr_scheduler = ExponentialLR(optimizer, gamma=0.99)
    elif self.args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=20)
    elif self.args.lr_scheduler == "LambdaLR":
        # sqrt: lr_lambda=lambda x: 1/np.sqrt(x) if x > 0 else 1
        # linear: lr_lambda=lambda x: 1 / x if x > 0 else 1
        # constant: lr_lambda=lambda x: 1
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1)
    elif self.args.lr_scheduler == "CyclicLR":
        lr_scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=0.1)

    return {
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'monitor': 'val_loss'
    }

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']    
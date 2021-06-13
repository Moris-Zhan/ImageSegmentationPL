import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pytorch_lightning as pl
import os
from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.utils import *
from LightningFunc.losses import configure_loss

from torchvision import models

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = int(in_channels / 2)
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class SegNet(pl.LightningModule):
    def __init__(self, num_classes, data_name):
        super(SegNet, self).__init__()
        self.num_classes = num_classes
        self.__build_model()
        self.__build_func(SegNet)       
        self.criterion = configure_loss('ce')

        self.checkname = self.backbone
        self.data_name = data_name
        self.dir = os.path.join("log_dir", self.data_name ,self.checkname) 
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)
        self.sample = (8, 3, 512, 256)
        self.sampleImg=torch.rand((1,3, 512, 256)).cuda()

    def __build_model(self):
        vgg = models.vgg19_bn(pretrained=True)
        # if pretrained:
        #     vgg.load_state_dict(torch.load(vgg19_bn_path))
        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:27])
        self.enc4 = nn.Sequential(*features[27:40])
        self.enc5 = nn.Sequential(*features[40:])

        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] +
              [nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True)] * 4)
        )
        self.dec4 = _DecoderBlock(1024, 256, 4)
        self.dec3 = _DecoderBlock(512, 128, 4)
        self.dec2 = _DecoderBlock(256, 64, 2)
        self.dec1 = _DecoderBlock(128, self.num_classes, 2)
        initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)

    def __build_func(self, obj):
        """Define model layers & loss."""

        self.backbone = "SegNet"
        setattr(obj, "training_step", training_step)
        setattr(obj, "training_epoch_end", training_epoch_end)
        setattr(obj, "validation_step", validation_step)
        setattr(obj, "validation_epoch_end", validation_epoch_end)
        setattr(obj, "test_step", test_step)
        setattr(obj, "test_epoch_end", test_epoch_end)
        setattr(obj, "configure_optimizers", configure_optimizers)
        setattr(obj, "prepare_matrix", prepare_matrix)   
        setattr(obj, "generate_matrix", generate_matrix)   
        setattr(obj, "saveDetail", saveDetail) 
        setattr(obj, "generate_score", generate_score)
        setattr(obj, "write_Best_model_path", write_Best_model_path)
        setattr(obj, "read_Best_model_path", read_Best_model_path) 

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([enc4, dec5], 1))
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))
        return dec1
if __name__ == '__main__':  
    # net = \
    # {
    #     "model":"smallunet",
    #     "drop_rate":0.3,
    #     "bn_momentum": 0.1 
    # }
    # input = \
    # {
    #     "data_type": "float32",
    #     "matrix_size": [160,160],
    #     "resolution": "0.15x0.15",
    #     "orientation": "RAI"
    # }

    # net = SegNet(nb_input_channels=3, n_classes=1,
    #              mean=0, std=0,
    #              orientation= "RAI",
    #              resolution= "0.15x0.15" ,
    #              matrix_size= [160,160],
    #              drop_rate= 0.3,
    #              bn_momentum= 0.1)
    net = SegNet(num_class = 21)
    print(net)
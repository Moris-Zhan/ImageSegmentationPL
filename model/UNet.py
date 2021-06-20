import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.utils import *
from LightningFunc.losses import configure_loss

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)     

class UNet(pl.LightningModule):
    def __init__(self, num_classes, args):
        super(UNet, self).__init__()
        self.n_channels = 3
        self.num_classes = num_classes
        self.bilinear = True
        self.__build_model()
        self.__build_func(UNet)   
        self.args = args    
        self.criterion = configure_loss(self.args.criterion)        

        self.checkname = self.backbone
        self.dir = os.path.join("log_dir", self.args.data_module ,self.checkname) 
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)
        self.sample = (8, 3, 512, 256)
        self.sampleImg=torch.rand((1,3, 512, 256)).cuda()

    def __build_model(self):
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = nn.Conv2d(64, self.num_classes, kernel_size=1) # OutConv

    def __build_func(self, obj):
        """Define model layers & loss."""

        self.backbone = "UNet"
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits










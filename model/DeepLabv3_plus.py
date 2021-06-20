import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
import os
from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.utils import *
from LightningFunc.losses import configure_loss
import model.backbone.DeepLabv3Plusbone as bones

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DeepLabv3_plus(pl.LightningModule):
    def __init__(self, num_classes, args):
            
        super(DeepLabv3_plus, self).__init__()
        self.num_classes = num_classes
        self.__build_model()
        self.__build_func(DeepLabv3_plus)       
        self.args = args    
        self.criterion = configure_loss(self.args.criterion)        

        self.checkname = self.backbone
        self.dir = os.path.join("log_dir", self.args.data_module ,self.checkname) 
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)
        self.sample = (8, 3, 512, 256)
        self.sampleImg=torch.rand((1,3, 512, 256)).cuda()

    def __build_model(self):
        nInputChannels=3
        n_classes=21
        os=16
        print("Constructing DeepLabv3+ model...")
        print("Number of classes: {}".format(n_classes))
        print("Output stride: {}".format(os))
        print("Number of Input Channels: {}".format(nInputChannels))

        # Atrous Conv
        self.resnet_features = bones.ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=True)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def __build_func(self, obj):
        """Define model layers & loss."""

        self.backbone = "DeepLabv3_plus"
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

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':    
    model = DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=True)    
    print(model)
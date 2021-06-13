
import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary
# import model.resnet as models
import model.backbone.PSPbone as bones
import pytorch_lightning as pl

from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.utils import *
from LightningFunc.losses import configure_loss

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)

class PSPNet(pl.LightningModule):
    def __init__(self, num_classes, data_name):
        super(PSPNet, self).__init__()
        self.layers = 50
        self.num_classes= num_classes
        self.__build_model()
        self.__build_func(PSPNet)
        self.data_name= data_name       
        self.criterion = configure_loss('ce')

        self.checkname = self.backbone
        self.data_name = data_name
        self.dir = os.path.join("log_dir", self.data_name ,self.checkname) 
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)
        self.sample = (8, 3, 512, 512)
        self.sampleImg=torch.rand((1,3, 512, 512)).cuda()

    def __build_model(self):
        pretrained=True
        if self.layers == 50:
            self.psp_size=2048
            self.deep_features_size=1024
            self.feats = bones.resnet50(pretrained=pretrained)
        elif self.layers == 101:
            self.psp_size=2048
            self.deep_features_size=1024
            self.feats = bones.resnet101(pretrained=pretrained)
        elif self.layers == 152:
            self.psp_size=2048
            self.deep_features_size=1024
            self.feats = bones.resnet152(pretrained=pretrained)

        self.psp = PSPModule(self.psp_size, 1024, (1, 2, 3, 6))
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, self.num_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )
    
    def __build_func(self, obj):
        """Define model layers & loss."""

        self.backbone = "PSPNet"
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
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        out = self.final(p)

        # auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
        # cls_out = self.classifier(auxiliary)

        return out

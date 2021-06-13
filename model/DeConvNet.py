import pytorch_lightning as pl

import torch.nn as nn
import torch
from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.utils import *
from LightningFunc.losses import configure_loss


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class DeConvNet(pl.LightningModule):
    def __init__(self, num_classes, data_name):
        super(DeConvNet, self).__init__()
        self.num_classes = num_classes
        self.__build_model()
        self.__build_func(DeConvNet)       
        self.criterion = configure_loss('ce')
        self.checkname = self.backbone
        self.data_name = data_name
        self.dir = os.path.join("log_dir", self.data_name ,self.checkname) 
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)
        self.sample = (8, 3, 512, 512)
        self.sampleImg=torch.rand((1,3, 512, 512)).cuda()
        

    def __build_model(self):
        self.input_size = 224
        self.name = 'DeconvNet'
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16, kernel_size=4,stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight) #Xaviers Initialisation
        self.swish1= nn.ReLU()

        #Max Pool 1
        self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.swish2 = nn.ReLU()

        #Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.swish3 = nn.ReLU()

        ##### SWITCH TO DECONVOLUTION PART #####

        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.swish4=nn.ReLU()

        #Max UnPool 1
        self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)

        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=5)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.swish5=nn.ReLU()

        #Max UnPool 2
        self.maxunpool2=nn.MaxUnpool2d(kernel_size=2)

        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=16,out_channels=self.num_classes,kernel_size=4)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.swish6=nn.Sigmoid()

    def __build_func(self, obj):
        """Define model layers & loss."""

        self.backbone = "DeConvNet"
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
        out=self.conv1(x)
        out=self.swish1(out)
        size1 = out.size()
        out,indices1=self.maxpool1(out)
        out=self.conv2(out)
        out=self.swish2(out)
        size2 = out.size()
        out,indices2=self.maxpool2(out)
        out=self.conv3(out)
        out=self.swish3(out)

        out=self.deconv1(out)
        out=self.swish4(out)
        out=self.maxunpool1(out,indices2,size2)
        out=self.deconv2(out)
        out=self.swish5(out)
        out=self.maxunpool2(out,indices1,size1)
        out=self.deconv3(out)
        out=self.swish6(out)
        return out
        
   
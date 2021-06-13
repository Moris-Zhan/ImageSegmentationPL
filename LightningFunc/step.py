import torch
from torch._C import dtype
from torchvision.utils import save_image

from LightningFunc.utils import *
from LightningFunc.optimizer import get_lr
import numpy as np
import cv2

def training_step(self, batch, batch_idx):
    # training_step defined the train loop.
    # It is independent of forward
    x, y = batch
    self.reference_image = x
    out = self.forward(x)
    loss = self.criterion(out, y) 
    
    # acc
    predicted = np.argmax(out.clone().detach().data.cpu().numpy(), axis=1)
    self.prepare_matrix(y.clone().detach().cpu().numpy(), predicted)

    values = {'Loss/Train': loss}
    self.log_dict(values, logger=True, on_epoch=False)
    self.logger.experiment.add_scalars("Loss/Step", {"Train":loss}, self.global_step)

    return {'loss':loss}

    

def training_epoch_end(self, outputs): # 在Validation的一個Epoch結束後，計算平均的Loss及Acc.
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    avg_train_Acc, avg_train_Acc_class, avg_train_mIoU, avg_train_FWIoU = self.generate_score()
    self.confusion_matrix = np.zeros((self.num_classes,) * 2)   

    self.logger.experiment.add_scalars("Loss/Epoch", {"Train":avg_loss}, self.current_epoch)
    self.logger.experiment.add_scalars("Accuracy/Acc_normal", {"Train":avg_train_Acc}, self.current_epoch)  
    self.logger.experiment.add_scalars("Accuracy/Acc_class", {"Train":avg_train_Acc_class}, self.current_epoch)  
    self.logger.experiment.add_scalars("Accuracy/mIoU", {"Train":avg_train_mIoU}, self.current_epoch)  
    self.logger.experiment.add_scalars("Accuracy/FWIoU", {"Train":avg_train_FWIoU}, self.current_epoch)
    # opt = self.optimizers()
    # self.logger.experiment.add_scalar("epoch/LR",
    #                                 get_lr(opt),
    #                                 self.current_epoch)

    if(self.current_epoch==1):    
        self.logger.experiment.add_graph(self, self.sampleImg)

    # iterating through all parameters
    for name,params in self.named_parameters():       
        self.logger.experiment.add_histogram(name,params,self.current_epoch)
    
def validation_step(self, batch, batch_idx):
    x, y = batch
    out = self.forward(x)
    loss = self.criterion(out, y) 

    # acc
    predicted = np.argmax(out.clone().detach().data.cpu().numpy(), axis=1)
    self.prepare_matrix(y.clone().detach().cpu().numpy(), predicted)

    # show result
    pred_rgb = decode_seg_map_sequence(predicted, self.data_name)   

    orign_img = np.array(x[0].cpu(), dtype=np.float64)   
    orign_img = orign_img.transpose((1, 2, 0))
    orign_img *= (0.229, 0.224, 0.225)
    orign_img += (0.485, 0.456, 0.406)
    orign_img *= 255.0  
    orign_img = cv2.resize(orign_img, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)

    # ratio = 0.7
    ratio = 0.0
    pred_rgb = pred_rgb[0].cpu().numpy().transpose((1, 2, 0))*255
    pred_rgb = cv2.resize(pred_rgb, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
    pred_rgb = np.add(orign_img*ratio , pred_rgb*(1 - ratio))
    values = {'Loss/Val':loss}
    self.log_dict(values, logger=True, on_epoch=True)
    self.logger.experiment.add_scalars("Loss/Step", {"Val":loss}, self.global_step)

    return {'loss':loss, "orign_img": torch.from_numpy(orign_img/255).type(torch.FloatTensor), \
            "pred_img":torch.from_numpy(pred_rgb/255).type(torch.FloatTensor) }
            


def validation_epoch_end(self, outputs): # 在Validation的一個Epoch結束後，計算平均的Loss及Acc.
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    avg_Val_Acc, avg_Val_Acc_class, avg_Val_mIoU, avg_Val_FWIoU = self.generate_score()
    self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    self.logger.experiment.add_scalars("Loss/Epoch", {"Val":avg_loss}, self.current_epoch)
    self.logger.experiment.add_scalars("Accuracy/Acc_normal", {"Val":avg_Val_Acc}, self.current_epoch)  
    self.logger.experiment.add_scalars("Accuracy/Acc_class", {"Val":avg_Val_Acc_class}, self.current_epoch)  
    self.logger.experiment.add_scalars("Accuracy/mIoU", {"Val":avg_Val_mIoU}, self.current_epoch)  
    self.logger.experiment.add_scalars("Accuracy/FWIoU", {"Val":avg_Val_FWIoU}, self.current_epoch)  


    orign_imgs = torch.stack([x['orign_img'] for x in outputs[:4]])
    orign_imgs = orign_imgs[:,:,:,[2,1,0]]
    self.logger.experiment.add_image("validation_%s/orign_img" %(self.current_epoch),
                                        orign_imgs,
                                        self.current_epoch,
                                        dataformats="NHWC")

    pred_imgs = torch.stack([x['pred_img'] for x in outputs[:4]])
    pred_imgs = pred_imgs[:,:,:,[2,1,0]]
    self.logger.experiment.add_image("validation_%s/pred_img" %(self.current_epoch),
                                        pred_imgs,
                                        self.current_epoch,
                                        dataformats="NHWC")

    self.write_Best_model_path()  

def test_step(self, batch, batch_idx): #定義 Test 階段
    x, y = batch
    out = self.forward(x)

    # acc
    predicted = np.argmax(out.clone().detach().data.cpu().numpy(), axis=1)
    self.prepare_matrix(y.clone().detach().cpu().numpy(), predicted)

    # show result
    pred_rgb = decode_seg_map_sequence(predicted, self.data_name)   

    orign_img = np.array(x[0].cpu(), dtype=np.float64)    
    orign_img = orign_img.transpose((1, 2, 0))
    orign_img *= (0.229, 0.224, 0.225)
    orign_img += (0.485, 0.456, 0.406)
    orign_img *= 255.0  
    orign_img = cv2.resize(orign_img, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)

    # ratio = 0.7
    ratio = 0.0
    pred_rgb = pred_rgb[0].cpu().numpy().transpose((1, 2, 0))*255
    pred_rgb = cv2.resize(pred_rgb, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
    pred_rgb = np.add(orign_img*ratio , pred_rgb*(1 - ratio))    

    # orign_y = np.array(y.cpu(), dtype=np.float64)    
    # orign_y = orign_y.transpose((1, 2, 0))
    # orign_y *= 255.0 

    # write cv2 IMG
    # cv2.imwrite(os.path.join(self.dir, 'orign_img.png'.format(batch_idx)), orign_img)
    # cv2.imwrite(os.path.join(self.dir, 'orign_y.png'.format(batch_idx)), orign_y)
    # cv2.imwrite(os.path.join(self.dir, 'pred_rgb.png'.format(batch_idx)), pred_rgb)
    
    
    return {"orign_img": torch.from_numpy(orign_img/255).type(torch.FloatTensor), \
            "pred_img":torch.from_numpy(pred_rgb/255).type(torch.FloatTensor) }
        

def test_epoch_end(self, outputs): # 在test的一個Epoch結束後，計算平均的Loss及Acc.
    # if self.data_name not in ["Cityscapes"]:
    avg_Test_Acc, avg_Test_Acc_class, avg_Test_mIoU, avg_Test_FWIoU = self.generate_score()
    self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    self.logger.experiment.add_scalars("Test/Acc_normal", {"":avg_Test_Acc}, self.current_epoch)  
    self.logger.experiment.add_scalars("Test/Acc_class", {"":avg_Test_Acc_class}, self.current_epoch)  
    self.logger.experiment.add_scalars("Test/mIoU", {"":avg_Test_mIoU}, self.current_epoch)  
    self.logger.experiment.add_scalars("Test/FWIoU", {"":avg_Test_FWIoU}, self.current_epoch)  
 

    # logging reference image       
    orign_imgs = torch.stack([x['orign_img'] for x in outputs[:4]])
    orign_imgs = orign_imgs[:,:,:,[2,1,0]]
    self.logger.experiment.add_image("Test_%s/orign_img" %(self.current_epoch),
                                        orign_imgs,
                                        self.current_epoch,
                                        dataformats="NHWC")

    pred_imgs = torch.stack([x['pred_img'] for x in outputs[:4]])
    pred_imgs = pred_imgs[:,:,:,[2,1,0]]
    self.logger.experiment.add_image("Test_%s/pred_img" %(self.current_epoch),
                                        pred_imgs,
                                        self.current_epoch,
                                        dataformats="NHWC")



    
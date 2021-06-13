# import os
# from argparse import ArgumentParser

from unicodedata import name

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, GPUStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os

from model.DeConvNet import DeConvNet
from model.PSPNet import PSPNet
from model.UNet import UNet
from model.FCN import FCN16s, FCN32s, FCN8s, FCNs
from model.FPN import FPN
from model.SegNet import SegNet
from model.DeepLabV3 import DeepLabV3
from model.DeepLabv3_plus import DeepLabv3_plus

from dataset.Cityscapes import CityscapeModule
from dataset.Pascal import VOCModule
from dataset.Coco import COCOModule
from dataset.BDD100K import BDD100KModule

if __name__ == '__main__':
    # dm = CityscapeModule(bsz=2)
    # dm = VOCModule(bsz=2)
    # dm = COCOModule(bsz=2)
    dm = BDD100KModule(bsz=2)
    dm.setup('fit')  

    
    # model = DeConvNet(dm.get_classes(), dm.name)
    # model = PSPNet(dm.get_classes(), dm.name)
    model = UNet(dm.get_classes(), dm.name)
    # model = FCN32s(dm.get_classes(), dm.name)
    # model = FPN(dm.get_classes(), dm.name)
    # model = SegNet(dm.get_classes(), dm.name)
    # model = DeepLabV3(dm.get_classes(), dm.name)
    # model = DeepLabv3_plus(dm.get_classes(), dm.name)

    setattr(model, "learning_rate", 1e-3)
    model.read_Best_model_path()

    root_dir = os.path.join('log_dir',dm.name)
    logger = TensorBoardLogger(root_dir, name= model.checkname, default_hp_metric =False )

    checkpoint_callback = ModelCheckpoint(
        monitor='Loss/Val', 
        dirpath= os.path.join(root_dir, model.checkname), 
        filename= model.checkname + '-{epoch:02d}-{Loss/Val:.2f}',
        save_top_k=3,
        mode='min',
        verbose=True
    )
    setattr(model, "checkpoint_callback", checkpoint_callback)
    
    early_stop_callback = EarlyStopping(
        monitor='Loss/Val',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    gpu_stats = GPUStatsMonitor() 

    trainer = Trainer(max_epochs = 2, gpus=-1, auto_select_gpus=True,precision=16,
                    logger=logger, num_sanity_val_steps=0, 
                    # weights_summary='full', 
                    auto_scale_batch_size = 'power', # only open when find batch_num
                    auto_lr_find=True,
                    accumulate_grad_batches=8,  # random error with (could not broadcast input array from shape)
                    callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, gpu_stats],
                    limit_train_batches=100,
                    limit_val_batches=100,
                    limit_test_batches = 100
                    )
                    
    # trainer.tune(model, datamodule=dm)
    trainer.fit(model, datamodule=dm)

    dm.setup('test') 
    trainer.test(model, datamodule=dm)

    # tensorboard --logdir=D:\WorkSpace\JupyterWorkSpace\ImageSegmentationPL\log_dir --samples_per_plugin images=100
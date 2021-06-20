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

import yaml
import argparse

def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in config.items():
            if isinstance(value, dict):
                for inside_key, inside_key_value in value.items():
                    setattr(args, inside_key, inside_key_value)
            else:
                setattr(args, key, value)
    return args

def load_data(args):
    dm = None
    if args.data_module == "CityscapeModule": dm = CityscapeModule(batch_size= args.batch_size)
    elif args.data_module == "VOCModule": dm = VOCModule(batch_size= args.batch_size)
    elif args.data_module == "COCOModule": dm = COCOModule(batch_size= args.batch_size)
    elif args.data_module == "BDD100KModule": dm = BDD100KModule(batch_size= args.batch_size)
    dm.setup(args.stage)
    return dm

def load_model(args, dm):
    model = None
    if args.model_name == "DeConvNet": model = DeConvNet(dm.get_classes(), args)
    elif args.model_name == "PSPNet": model = PSPNet(dm.get_classes(), args)
    elif args.model_name == "UNet": model = UNet(dm.get_classes(), args)
    elif args.model_name == "FCN32s": model = FCN32s(dm.get_classes(), args)
    elif args.model_name == "FPN": model = FPN(dm.get_classes(), args)
    elif args.model_name == "SegNet": model = SegNet(dm.get_classes(), args)
    elif args.model_name == "DeepLabV3": model = DeepLabV3(dm.get_classes(), args)
    elif args.model_name == "DeepLabv3_plus": model = DeepLabv3_plus(dm.get_classes(), args)

    return model

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/config.yaml',
            help='YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args)
    dm = load_data(args)
    model = load_model(args, dm)

    # setattr(model, "learning_rate", 1e-3)
    model.read_Best_model_path()

    root_dir = os.path.join('log_dir',dm.name)
    logger = TensorBoardLogger(root_dir, name= model.checkname, default_hp_metric =False )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', 
        dirpath= os.path.join(root_dir, model.checkname), 
        filename= model.checkname + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        verbose=True
    )
    setattr(model, "checkpoint_callback", checkpoint_callback)
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    gpu_stats = GPUStatsMonitor() 

    trainer = Trainer.from_argparse_args(config, 
                                        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, gpu_stats],
                                        logger=logger)
                    
    if args.tune:
        trainer.tune(model, datamodule=dm)

    trainer.fit(model, datamodule=dm)

    dm.setup('test') 
    trainer.test(model, datamodule=dm)

    # tensorboard --logdir=D:\WorkSpace\JupyterWorkSpace\ImageSegmentationPL\log_dir --samples_per_plugin images=100
from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import cv2
import albumentations as A
from matplotlib import pyplot as plt
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from copy import deepcopy
import torch

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    if image.shape[-1] != 3: 
        image = np.transpose(image, (1, 2, 0))

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()

# -----------------------------------------------------------------------------


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,                 
                 split='trainval',
                 year = '2007'
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = 'D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\VOC\\VOCdevkit\\VOC{}'.format(year)
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        self.year = year
        self.split = split                   
        self.num_classes = 21

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        with open(os.path.join(os.path.join(_splits_dir, self.split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(self._image_dir, line + ".jpg")
            _cat = os.path.join(self._cat_dir, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = cv2.imread(self.images[index], cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.categories[index])
        mask = mask[:,:,[2,1,0]]
        mask = self.encode_segmap(mask)
        return (image, mask)    

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def get_pascal_labels(self):
    	return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                           [0, 64, 128]])

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.num_classes = 21
        self.view_mark = False

    def __getitem__(self, index):
        x, y = self.subset[index]

        transformed = self.transform(image=x, mask=y)
        image, mask = transformed["image"], transformed["mask"]

        if self.view_mark:
            visualize(image, mask) 

        return image, mask        


    def __len__(self):
        return len(self.subset)

class VOCModule(pl.LightningDataModule):
    def __init__(self, batch_size, base_size = 256, crop_size = 64):
        super().__init__()
        base_size = 512
        crop_size = 512
        self.batch_size = batch_size
        self.base_size = base_size
        self.crop_size = crop_size
        self.name = "VOC"

        # year = 2007 or 2012
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            train_set = VOCSegmentation(split='train', year='2007')
            val_set = VOCSegmentation(split='val', year='2007')            

            self.train_dataset = DatasetFromSubset(
                train_set, 
                transform = A.Compose([
                        A.RandomResizedCrop(self.crop_size, self.crop_size, p=1),
                        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.2),
                        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.2),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2(),
                    ])
            )
            self.val_dataset = DatasetFromSubset(
                val_set, 
                transform = A.Compose([
                    A.Resize(self.base_size, self.base_size, p=1),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            )

            self.num_classes = train_set.num_classes
        if stage == 'test' or stage is None:
            self.base_size, self.crop_size = 512, 512
            self.test_dataset = VOCSegmentation(split='test', year='2007')
            self.test_dataset = DatasetFromSubset(
                self.test_dataset, 
                transform = A.Compose([
                    A.Resize(self.base_size, self.crop_size, p=1),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            )

    def prepare_data(self):
        # 在该函数里一般实现数据集的下载等，只有cuda:0 会执行该函数
        pass
    def train_dataloader(self):
        return DataLoader(
                dataset=self.train_dataset,# TensorDataset类型数据集
                batch_size=self.batch_size,# mini batch size
                shuffle=True,# 设置随机洗牌
                num_workers=5# 加载数据的进程个数
            )

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        return DataLoader(
                dataset=self.val_dataset,# TensorDataset类型数据集
                batch_size=self.batch_size,# mini batch size
                shuffle=False,# 设置随机洗牌
                num_workers=5# 加载数据的进程个数
            )

    def test_dataloader(self):
        return DataLoader(
                dataset=self.test_dataset,# TensorDataset类型数据集
                batch_size=1,# mini batch size
                shuffle=False,# 设置随机洗牌
                num_workers=5# 加载数据的进程个数
            )
    def get_classes(self):
        return self.num_classes
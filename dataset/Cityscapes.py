import os
import numpy as np
import scipy.misc as m
from torch.utils import data
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
    # if mask.shape[-1] != 3: 
    #     mask = np.transpose(mask, (0, 1))

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

class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 19

    def __init__(self, split="train"):
        super().__init__()
        '''
        Cityscapes是關於城市街道場景的語義理解圖片數據集。它主要包含來自50個不同城市的街道場景，
        擁有5000張在城市環境中駕駛場景的高質量像素級註釋圖像（其中2975 for train，500 for val, 1525 for test， 共有19個類別）；
        測試集(test)只給了原圖，沒有給標籤
        此外，它還有20000張粗糙標註的圖像(gt coarse)。
        從我目前了解來說， 一般都是拿這5000張精細標註(gt fine)的樣本集來進行訓練和評估的
      


        xxx_color.png是標註的可視化圖片

        # xxx_labelsIds.png是語義分割訓練需要的。它們的像素值就是class值
        xxx_instanceIds.png是用來做實例分割訓練用的
        xxx_polygons.json是用labelme工具標註後所生成的文件
        '''

        self.root = 'D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Segmentation\\Cityscapes'
        self.split = split
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split) 
        # gtFine，顯然這裡的fine就是精細標註的意思。gtFine下面也是分為train， test以及val，然後它們的子目錄也是以城市為單位來放置圖片

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png') 
                                # os.path.basename(img_path)[:-15] + 'gtFine_color.png') 


        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        mask = self.encode_segmap(mask)
        return (image, mask)

    def encode_segmap(self, mask): # preprocess_mask
        # Put all void classes to zero
        mask = mask.astype(np.float32)
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        # mask[mask == 2.0] = 0.0
        # mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
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

class CityscapeModule(pl.LightningDataModule):
    def __init__(self, bsz, base_size = 128, crop_size = 64):
        super().__init__()
        base_size = 512
        crop_size = 512
        self.batch_size = bsz
        self.base_size = base_size
        self.crop_size = crop_size
        self.name = "Cityscapes"


    def setup(self, stage):
        if stage == 'fit' or stage is None:
            full_dataset = CityscapesSegmentation(split='train')
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_set, val_set = torch.utils.data.dataset.random_split(full_dataset, [train_size, test_size])
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
            self.num_classes = len(full_dataset.valid_classes)

        if stage == 'test' or stage is None:
            self.base_size, self.crop_size = 1024, 512
            test_set = CityscapesSegmentation(split='test')
            self.test_dataset = DatasetFromSubset(
                test_set, 
                transform = A.Compose([
                    A.Resize(self.base_size, self.crop_size, p=1),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            )
            self.num_classes = len(test_set.valid_classes)
    def prepare_data(self):
        # 在该函数里一般实现数据集的下载等，只有cuda:0 会执行该函数
        pass
    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
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
        # return DataLoader(self.test_dataset, batch_size=1, shuffle=True)
        return DataLoader(
                dataset=self.test_dataset,# TensorDataset类型数据集
                batch_size=1,# mini batch size
                shuffle=False,# 设置随机洗牌
                num_workers=5# 加载数据的进程个数
            )
    def get_classes(self):
        return self.num_classes
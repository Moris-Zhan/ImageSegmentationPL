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
from glob import glob

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


class BDD100KSegmentation(Dataset):
    """
    BDD100K dataset
    """
    classes = [
        "unlabeled",
        "dynamic",
        "ego_vehicle",
        "ground",
        "static",
        "parking",
        "rail track",
        "road",
        "sidewalk",
        "bridge",
        "building",
        "fence",
        "garage",
        "guard rail",
        "tunnel",
        "wall",
        "banner",
        "billboard",
        "lane divider",
        "parking_sign",
        "pole",
        "polegroup",
        "street_light",
        "traffic_cone",
        "traffic_device",
        "traffic_light",
        "traffic_sign",
        "traffic_sign_frame",
        "terrain",
        "vegetation",
        "sky",
        "person",
        "rider",
        "bicycle",
        "bus",
        "car",
        "caravan",
        "motorcycle",
        "trailer",
        "train",
        "truck"
    ]
    def __init__(self,                 
                 split,
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = 'D:\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Segmentation\\bdd100k\\bdd100k'
        self._image_dir = os.path.join(self._base_dir, 'images','10k',split)
        self._cat_dir = os.path.join(self._base_dir, 'labels','sem_seg','colormaps',split)
        self.split = split                   
        self.num_classes = len(self.classes)

        self.im_ids = []
        self.images = []
        self.categories = []
        if split != "test":
            for label_mask in glob(os.path.join(self._cat_dir, "*.png")):
                fn = os.path.basename(label_mask)
                path = os.path.join(self._image_dir, "%s" % (fn) ).replace("png","jpg")
                self.images.append(path)
                self.categories.append(label_mask)
        else:
            for path in glob(os.path.join(self._image_dir, "*.jpg")):
                fn = os.path.basename(path)
                self.images.append(path)
                self.categories.append("")

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = cv2.imread(self.images[index], cv2.COLOR_BGR2RGB)
        if self.categories[index] != "":
            mask = cv2.imread(self.categories[index])
            mask = mask[:,:,[2,1,0]]
            mask = self.encode_segmap(mask)
            return (image, mask)
        else: 
            h, w, _ = image.shape
            mask = np.zeros((h, w), dtype=np.float)
            return (image, mask) 
    

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
        for ii, label in enumerate(self.get_bdd100k_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def get_bdd100k_labels(self):
        unlabeled = [0, 0, 0]
        dynamic = [111, 74, 0]
        ego_vehicle = [0, 0, 0]
        ground = [81, 0, 81]
        static = [0, 0, 0]
        parking = [250, 170, 160]
        rail_track = [230, 150, 140]
        road = [128, 64, 128]
        sidewalk = [244, 35, 232]
        bridge = [150, 100, 100]
        building = [70, 70, 70]
        fence = [190, 153, 153]
        garage = [180, 100, 180]
        guard_rail = [180, 165, 180]
        tunnel = [150, 120, 90]
        wall = [102, 102, 156]
        banner = [250, 170, 100]
        billboard = [220, 220, 250]
        lane_divider = [255, 165, 0]
        parking_sign = [220, 20, 60]
        pole = [153, 153, 153]
        polegroup = [153, 153, 153]
        street_light = [220, 220, 100]
        traffic_cone = [255, 70, 0]
        traffic_device = [220, 220, 220]
        traffic_light = [250, 170, 30]
        traffic_sign = [220, 220, 0]
        traffic_sign_frame = [250, 170, 250]
        terrain = [152, 251, 152]
        vegetation = [107, 152, 35]
        sky = [70, 130, 180]
        person = [220, 20, 60]
        rider = [255, 0, 0]
        bicycle = [119, 11, 32]
        bus = [0, 60, 100]
        car = [0, 0, 142]
        caravan = [0, 0, 90]
        motorcycle = [0, 0, 230]
        trailer = [0, 0, 110]
        train = [0, 80, 100]
        truck = [0, 0, 70]

        return np.array([
            unlabeled, dynamic, ego_vehicle, ground, static, parking, rail_track, road, sidewalk, bridge,
            building, fence, garage, guard_rail, tunnel, wall, banner, billboard, lane_divider, parking_sign,
            pole, polegroup, street_light, traffic_cone, traffic_device, traffic_light, traffic_sign,
            traffic_sign_frame, terrain, vegetation, sky, person, rider, bicycle, bus, car, caravan,
            motorcycle, trailer, train, truck])

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

class BDD100KModule(pl.LightningDataModule):
    def __init__(self, batch_size, base_size = 256, crop_size = 64):
        super().__init__()
        base_size = 512
        crop_size = 512
        self.batch_size = batch_size
        self.base_size = base_size
        self.crop_size = crop_size
        self.name = "BDD100K"

        # year = 2007 or 2012
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            train_set = BDD100KSegmentation(split='train')
            val_set = BDD100KSegmentation(split='val')            

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
            test_set = BDD100KSegmentation(split='test')
            self.test_dataset = DatasetFromSubset(
                test_set, 
                transform = A.Compose([
                    A.Resize(self.base_size, self.crop_size, p=1),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            )
            self.num_classes = test_set.num_classes

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
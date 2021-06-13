import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import trange
import os
import cv2
from pycocotools.coco import COCO
from pycocotools import mask
from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import cv2
import albumentations as A
from matplotlib import pyplot as plt
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import torch
from glob import glob

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    if image.shape[-1] != 3: 
        image = np.transpose(image, (1, 2, 0))

    image = image[:,:,[2,1,0]]
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

class COCOSegmentation(Dataset):
    NUM_CLASSES = 21
    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask
    
    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width']) 
            target_path = ""
            if self.year == '2014':          
                target_path = "/home/eagleuser/Users/leyan/coco/mask/{}{}/COCO_{}{}_{}.png".format(self.split, self.year, self.split, self.year, str(img_id).zfill(12))
            elif self.year == '2017': 
                target_path = "/home/eagleuser/Users/leyan/coco/mask/{}{}/{}.png".format(self.split, self.year, str(img_id).zfill(12))               
            
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
                plt.imsave(target_path, mask)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
            # break                                 
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def __init__(self,
                 split='train',
                 year='2017'):
        super().__init__()
        """
        NUM_CHANNEL = 91
        [] background
        [5] airplane
        [2] bicycle
        [16] bird
        [9] boat
        [44] bottle
        [6] bus
        [3] car
        [17] cat
        [62] chair
        [21] cow
        [67] dining table
        [18] dog
        [19] horse
        [4] motorcycle
        [1] person
        [64] potted plant
        [20] sheep
        [63] couch
        [7] train
        [72] tv
        """
        CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
        base_dir = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\COCO"
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year))
        mask_folder = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\COCO\\mask\\{}{}\\".format(split, year)
        self.num_classes = CAT_LIST

        self.split = split
        self.year = year
        if self.split != 'test':        
            self.coco = COCO(ann_file)
            self.coco_mask = mask
            self.mask_folder = mask_folder
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)
            if os.path.exists(ids_file):
                self.ids = torch.load(ids_file)
            else:
                ids = list(self.coco.imgs.keys())
                self.ids = self._preprocess(ids, ids_file)
            # Display stats
            print('Number of images in {}: {:d}'.format(split, len(ann_file)))
        else:
            self.ids = glob(os.path.join(base_dir, "images\\{}{}".format(split, year),"*.jpg"))
            print('Number of images in {}: {:d}'.format(split, len(self.ids)))
        

    def __getitem__(self, index):        
        img_id = self.ids[index]
        if 'test' not in img_id:
            coco = self.coco
            img_metadata = coco.loadImgs(img_id)[0]
            path = img_metadata['file_name']

            img_path = os.path.join(self.img_dir, path)
            target_path = os.path.join(self.mask_folder, path.replace("jpg", "png"))

            image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
            mask = Image.open(target_path).convert('P')
            mask = np.array(mask)
            return (image, mask) 
        else:
            img_path = self.ids[index]
            image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

            h, w, _ = image.shape
            mask = np.zeros((h, w), dtype=np.float)
            return (image, mask) 

    def __len__(self):
        return len(self.ids)

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.num_classes = 21
        self.view_mark = False

    def __getitem__(self, index):
        x, y = self.subset[index]

        transformed = self.transform(image=x, mask=y) # operands could not broadcast input array from shape (512,1) into shape (512)
        image, mask = transformed["image"], transformed["mask"]
        # image = image.reshape(3,512,512)
        # mask = mask.reshape(512,512)

        if self.view_mark:
            visualize(image, mask) 

        return image, mask        


    def __len__(self):
        return len(self.subset)

class COCOModule(pl.LightningDataModule):
    def __init__(self, bsz, base_size = 256, crop_size = 64):
        super().__init__()
        base_size = 512
        crop_size = 512
        self.batch_size = bsz
        self.base_size = base_size
        self.crop_size = crop_size
        self.name = "Coco"
        self.year='2017'

        # year = 2014 or 2017
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            full_dataset = COCOSegmentation(split='train', year=self.year)
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

            self.num_classes = len(full_dataset.num_classes)
        if stage == 'test' or stage is None:
            self.base_size, self.crop_size = 512, 512
            test_set = COCOSegmentation(split='test', year=self.year)
            self.test_dataset = DatasetFromSubset(
                test_set, 
                transform = A.Compose([
                    A.Resize(self.base_size, self.crop_size, p=1),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            )
            self.num_classes = len(test_set.num_classes)
    def prepare_data(self):
        # 在该函数里一般实现数据集的下载等，只有cuda:0 会执行该函数
        pass
    def train_dataloader(self):
        return DataLoader(
                dataset=self.train_dataset,# TensorDataset类型数据集
                batch_size=self.batch_size,# mini batch size
                shuffle=True,# 设置随机洗牌
                num_workers=0# 加载数据的进程个数
            )

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        return DataLoader(
                dataset=self.val_dataset,# TensorDataset类型数据集
                batch_size=self.batch_size,# mini batch size
                shuffle=False,# 设置随机洗牌
                num_workers=0# 加载数据的进程个数
            )

    def test_dataloader(self):
        return DataLoader(
                dataset=self.test_dataset,# TensorDataset类型数据集
                batch_size=1,# mini batch size
                shuffle=False,# 设置随机洗牌
                num_workers=0# 加载数据的进程个数
            )
    def get_classes(self):
        return self.num_classes
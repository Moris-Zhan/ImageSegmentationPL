import numpy as np


def prepare_matrix(self, gt_image, pre_image):
    for lp, lt in zip(pre_image, gt_image):
        self.confusion_matrix += self.generate_matrix(self.num_classes, lt.flatten(), lp.flatten())    

def generate_matrix(self, C, gt_image, pre_image):
    mask = (gt_image >= 0) & (gt_image < C)
    label = C * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=C**2)[:C**2]
    confusion_matrix = count.reshape(C, C)
    return confusion_matrix    

def generate_score(self):
    Acc, Acc_class, mIoU, FWIoU = Pixel_Accuracy(self.confusion_matrix), \
                                    Pixel_Accuracy_Class(self.confusion_matrix), \
                                    Mean_Intersection_over_Union(self.confusion_matrix), \
                                    Frequency_Weighted_Intersection_over_Union(self.confusion_matrix)
    return Acc, Acc_class, mIoU, FWIoU

def Pixel_Accuracy(confusion_matrix):
    Acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return Acc

def Pixel_Accuracy_Class(confusion_matrix):
    Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    Acc = np.nanmean(Acc)
    return Acc


def Mean_Intersection_over_Union(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU


def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))

    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

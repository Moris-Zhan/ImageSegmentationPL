import torch
import matplotlib.image as img 

import cv2
import dlib
from imutils.face_utils import *

import numpy as np

# image = img.imread("extra//test.jpg")
# image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # opencvImage
dlib_path = 'extra//shape_predictor_68_face_landmarks.dat'

def get_face(img):
    global detector, landmark_predictor
    # 宣告臉部偵測器，以及載入預訓練的臉部特徵點模型
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(dlib_path)
    # 產生臉部識別
    face_rects = detector(img, 1)    
    for i, d in enumerate(face_rects):
        # 讀取框左上右下座標
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        # 根據此座標範圍讀取臉部特徵點
        shape = landmark_predictor(img, d)
        # 將特徵點轉為numpy
        shape = shape_to_np(shape)  # (68,2)
        # 透過dlib挖取臉孔部分，將臉孔圖片縮放至256*256的大小，並存放於pickle檔中
        # 人臉圖像部分呢。很簡單，只要根據畫框的位置切取即可crop_img = img[y1:y2, x1:x2, :]
        crop_img = img[y1:y2, x1:x2, :]
        try:
            resize_img = cv2.resize(crop_img, (512, 512))
            # cv2.imshow("OpenCV",resize_img) 
            # cv2.waitKey()
            return resize_img
        except:
            return np.array([0])
    return np.array([0])

def predict_image(image, model):
    try:
        face = get_face(image) # predict target
        face = torch.tensor(face, dtype=torch.float32)/255 # normalize
        face = face.permute(2, 0, 1).unsqueeze(0).cuda()
        # model = torch.load('run\SCUT\pre_googlenet\experiment_6\pre_googlenet.pkl')
        # model.load_state_dict(torch.load('run\SCUT\pre_googlenet\experiment_6\checkpoint.pth.tar')['state_dict'])
        outputs = model(face) # [batch_size, c, h, w]
        _, predicted = torch.max(outputs.data, 1)
        score = int(predicted.item()) * 20
        return score
    except Exception as e:
        # print(e)
        return 0    
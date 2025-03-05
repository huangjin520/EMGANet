""""data augment
旋转————平移——————翻转——————
HSV   H:0.015  S:0.7  V:0.4"""

import pandas as pd
import numpy as np
import tqdm
import shutil
from glob import glob
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from natsort import natsorted
import cv2
from sklearn.model_selection import train_test_split
import random
import torch
import matplotlib.pyplot as plt
def Contrast(img_path,a = 1.5,clip=(20,200)):
    in_image = cv2.imread(img_path, 0)
    out_image = float(a) * in_image
    # 进行数据截断, 大于255的值要截断为255
    out_image[out_image < clip[0]] = 0
    out_image[out_image > clip[1]] = 255
    # 数据类型转化
    out_image = np.round(out_image)
    out_image = out_image.astype(np.uint8)
    # 显示原图像和线性变化后的结果
    return out_image

def Contrast_gamma(img_path,gamma = 0.8):
    in_image = cv2.imread(img_path, 0)
    fI = in_image / 255.0
    # 伽马变化
    out_image = np.power(fI, gamma) * 255  # 图像转化为0-255分布
    out_image = out_image.astype(np.uint8)
    return out_image

def Light(img_path,value):
    in_image = cv2.imread(img_path, 0)
    in_image = in_image.astype(np.int16)

    out_image = in_image + value
    out_image[out_image < 0] = 0
    out_image[out_image >255] = 255
    out_image = out_image.astype(np.uint8)
    return out_image


def Blur(img_path,blur):
    in_image = cv2.imread(img_path, 0)
    if blur ==1:
        out_image = cv2.blur(in_image, (5, 5))  #均值滤波
    if blur ==2:
        out_image = cv2.GaussianBlur(in_image, (7, 7),5)  # 高斯滤波
    if blur ==3:
        out_image = cv2.bilateralFilter(in_image, 10, 10, 50)  # 高斯滤波
    out_image = out_image.astype(np.uint8)
    return out_image


def make_dataset(root, datasets='train', test_size=0.1):
    dataset = []
    # print(root)

    dir_img = os.path.join(root, 'img')
    dir_gt = os.path.join(root, 'gt')

    img_list = natsorted(os.listdir(dir_img))

    for index in img_list:
        img_name = index.split('.')[0]
        img = index
        gt = img_name + '_mask.bmp'
        dataset.append([os.path.join(dir_img, img), os.path.join(dir_gt, gt)])
    if (test_size==0)|(test_size==1):
        shuffle =True
        random.seed(123)
        if shuffle:
            random.shuffle(dataset)
        return dataset
    train_data, val_data = train_test_split(dataset, test_size=test_size, random_state=123, shuffle=True)
    # print(len(train_data),len(val_data))
    if datasets == 'train':
        dataset = train_data

    if datasets == 'val':
        dataset = val_data
    return dataset


"""jin 数据划分"""
# train_dataset = make_dataset(r"/mnt/Disk1/huangjin/breast_ultrasound/JIN_Work_algorithm/dataset_public/Dataset_BUSI_with_GT/BUSI_train_test/trian/img/ori/", datasets='train', test_size=1)
# val_dataset = make_dataset(r"../../datasets", datasets='val', test_size=0.2)
# print(len(train_dataset))
# print(len(val_dataset))

# for val_img , val_gt in val_dataset:
#     val_img_name = os.path.basename(val_img)
#     val_gt_name = os.path.basename(val_gt)
#     new_img_path = "valid/img/"+val_img_name
#     new_gt_path = "valid/gt/"+val_gt_name
#     print(new_img_path)
#     shutil.copyfile(val_img, new_img_path)
#     shutil.copyfile(val_gt, new_gt_path)


"""jin 后面全是数据增强"""
train_img_paths = glob("/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/img/*.bmp")
train_gt_paths  = glob("/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/mask/*.bmp")
print(len(train_img_paths), len(train_gt_paths))

for i in range(len(train_img_paths)):
    train_img_path, train_gt_path = train_img_paths[i] , train_gt_paths[i]
    train_img_name = os.path.basename(train_img_path)
    train_gt_name = os.path.basename(train_gt_path)
    # new_img_path = "train/img/" + train_img_name
    # new_gt_path = "train/gt/" + train_gt_name
    # print(new_img_path)
    # shutil.copyfile(train_img_path, new_img_path)
    # shutil.copyfile(train_gt_path, new_gt_path)

    """数据增强"""
    img = cv2.imread(train_img_path, 0)
    img1 = Contrast_gamma(train_img_path, gamma=0.6)
    img1_img_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/img", "Contrast1_"+train_img_name)
    img1_gt_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/mask", "Contrast1_"+ train_gt_name)
    cv2.imwrite(img1_img_path,img1)
    shutil.copyfile(train_gt_path, img1_gt_path)
    print(train_img_path, img1_img_path)


    img2 = Contrast_gamma(train_img_path, gamma=0.8)
    img2_img_path =os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/img", "Contrast2_" + train_img_name)
    img2_gt_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/mask", "Contrast2_" + train_gt_name)
    cv2.imwrite(img2_img_path,img2)
    shutil.copyfile(train_gt_path, img2_gt_path)


    img3 = Contrast_gamma(train_img_path, gamma=0.9)
    img3_img_path =  os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/img","Contrast3_" + train_img_name)
    img3_gt_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/mask", "Contrast3_" + train_gt_name)
    cv2.imwrite(img3_img_path,img3)
    shutil.copyfile(train_gt_path, img3_gt_path)


    img4 = Light(train_img_path, value=20)
    img4_img_path =  os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/img", "Light1_" + train_img_name)
    img4_gt_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/mask", "Light1_" + train_gt_name)
    cv2.imwrite(img4_img_path,img4)
    shutil.copyfile(train_gt_path, img4_gt_path)


    img5 = Light(train_img_path, value=50)
    img5_img_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/img", "Light2_" + train_img_name)
    img5_gt_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/mask", "Light2_" + train_gt_name)
    cv2.imwrite(img5_img_path,img5)
    shutil.copyfile(train_gt_path, img5_gt_path)


    img6 = Light(train_img_path, value=-10)
    img6_img_path =os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/img", "Light3_" + train_img_name)
    img6_gt_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/mask", "Light3_" + train_gt_name)
    cv2.imwrite(img6_img_path,img6)
    shutil.copyfile(train_gt_path, img6_gt_path)

    img7 = Blur(train_img_path, blur=1)
    img7_img_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/img", "Blur1_" + train_img_name)
    img7_gt_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/mask", "Blur1_" + train_gt_name)
    cv2.imwrite(img7_img_path,img7)
    shutil.copyfile(train_gt_path, img7_gt_path)

    img8 = Blur(train_img_path, blur=2)
    img8_img_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/img", "Blur2_" + train_img_name)
    img8_gt_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/mask", "Blur2_" + train_gt_name)
    cv2.imwrite(img8_img_path,img8)
    shutil.copyfile(train_gt_path, img8_gt_path)


    img9 = Blur(train_img_path, blur=3)
    img9_img_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/img", "Blur3_" + train_img_name)
    img9_gt_path = os.path.join(r"/mnt/Disk1/maoyz/BUSI_train_test/train/data_augment/mask", "Blur3_" + train_gt_name)
    cv2.imwrite(img9_img_path,img9)
    shutil.copyfile(train_gt_path, img9_gt_path)


#     fig,ax = plt.subplots(2, 5)
#
#     ax[0,0].imshow(img, plt.cm.gray)
#     ax[0,0].set_title("img")
#
#     ax[0,1].imshow(img1, plt.cm.gray)
#     ax[0,1].set_title(os.path.basename(img1_img_path))
#
#     ax[0,2].imshow(img2, plt.cm.gray)
#     ax[0,2].set_title(os.path.basename(img2_img_path))
#
#     ax[0,3].imshow(img3, plt.cm.gray)
#     ax[0,3].set_title(os.path.basename(img3_img_path))
#
#     ax[0,4].imshow(img4, plt.cm.gray)
#     ax[0,4].set_title(os.path.basename(img4_img_path))
#
#     ax[1,0].imshow(img5, plt.cm.gray)
#     ax[1,0].set_title(os.path.basename(img5_img_path))
#
#     ax[1,1].imshow(img6, plt.cm.gray)
#     ax[1,1].set_title(os.path.basename(img6_img_path))
#
#     ax[1,2].imshow(img7, plt.cm.gray)
#     ax[1,2].set_title(os.path.basename(img7_img_path))
#
#     ax[1,3].imshow(img8, plt.cm.gray)
#     ax[1,3].set_title(os.path.basename(img8_img_path))
#
#     ax[1,4].imshow(img9, plt.cm.gray)
#     ax[1,4].set_title(os.path.basename(img9_img_path))
#
#
#
#
#
#
#
#     # plt.get_current_fig_manager().full_screen_toggle()
#     figManager = plt.get_current_fig_manager()
#     figManager.window.showMaximized()
# #
# #
#
#     plt.show()
#     # plt.pause(10) #交互模式和阻塞模式
#     plt.close()
#
#






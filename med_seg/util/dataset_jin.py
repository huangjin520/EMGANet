import torch
from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
import numpy as np
from PIL import Image
import random, os
import pandas as pd



class CrackData(Dataset):
    def __init__(self, df, transforms=None, img_size= False):
        self.data = df
     
        self.transform = transforms
        self.img_size = img_size
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data['images'].iloc[idx]).convert("RGB")   #change color model "L", "RGB"
        gt = Image.open(self.data['masks'].iloc[idx]).convert('L')

        img_shape = gt.size      #补充的图片尺寸属性  ---JIN
        w, h = img_shape
        image_path = self.data['images'].iloc[idx]
        gt_path = self.data['masks'].iloc[idx]
        sample = {'image': img, 'gt': gt}
    
        sample = self.transform(sample)
        if self.img_size is True:
            return sample['image'], sample['gt'], w, h, image_path, gt_path
    
        return sample['image'], sample['gt']

class CrackData_data_augmentation_train(Dataset):
    """jin 用于交叉验证, 且支持离线数据的增强的模式"""
    def __init__(self, df, transforms=None, img_size= False):
        self.data_ori = df 
        self.look_augmentaion()
        df_all = pd.concat([self.data_ori, self.augmentation_all], axis=0)
        
        self.data = df_all
     
        self.transform = transforms
        self.img_size = img_size
    def look_augmentaion(self):
        image_paths = self.data_ori['images'].tolist()
        gt_paths = self.data_ori['masks'].tolist()
        
        image_names = [os.path.basename(i) for i in image_paths]
        
        image_augmentation_dir_path = os.path.dirname(image_paths[0]).replace("ori", "data_augment")
        gt_augmentation_dir_path = os.path.dirname(gt_paths[0]).replace("ori", "data_augment")
        
        image_augmentation_names = os.listdir(image_augmentation_dir_path)
        
        dst_image_augmentation_names = [i for i in image_augmentation_names if i.split('_')[1] in image_names]
        dst_gt_augmentation_names = [i.replace(".bmp", "_anno.bmp") for i in dst_image_augmentation_names]
        
        dst_image_augmentation_paths = [os.path.join(image_augmentation_dir_path, i ) for i in dst_image_augmentation_names]
        dst_gt_augmentation_paths = [os.path.join(gt_augmentation_dir_path, i ) for i in dst_gt_augmentation_names]
        
        self.augmentation_all = pd.DataFrame({'images': dst_image_augmentation_paths, 'masks': dst_gt_augmentation_paths})
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data['images'].iloc[idx]).convert("RGB")   #change color model "L", "RGB"
        gt = Image.open(self.data['masks'].iloc[idx]).convert('L')
        images_path = [self.data['images'].iloc[idx], self.data['masks'].iloc[idx]]

        img_shape = gt.size      #补充的图片尺寸属性  ---JIN
        w, h = img_shape
        image_path = self.data['images'].iloc[idx]
        gt_path = self.data['masks'].iloc[idx]
        
        sample = {'image': img, 'gt': gt}
    
        sample = self.transform(sample)
        if self.img_size is True:
            return sample['image'], sample['gt'], w, h, image_path, gt_path
    
        return sample['image'], sample['gt']









framObjTrain = {'img' : [],
           'mask' : []
          }



      



# if __name__=="__main__":
#     from transforms import *
#     from torch.utils.data import DataLoader
#     from glob import glob
#     transforms_train = transforms.Compose([
#                 RandomHorizontalFlip(),
#                 RandomVerticalFlip(),
#                 RandomRotate(30),
#                 Resize((448,384)),
#                           ToTensor()])
#     transforms_test =ToTensor()
#     transforms_val =transforms.Compose([
#         Resize((448, 384)),
#         ToTensor()])
#     train_path = "../dataset/train"
#     val_path = "../dataset/valid"
#     test_path = "../dataset/test"
#
#     train_data = pd.DataFrame({'images': sorted(glob(os.path.join(train_path, "img") + "/*.bmp")),
#                                'masks': sorted(glob(os.path.join(train_path, "gt") + "/*.bmp"))})
#
#     val_data = pd.DataFrame({'images': sorted(glob(os.path.join(val_path, "img") + "/*.bmp")),
#                              'masks': sorted(glob(os.path.join(val_path, "gt") + "/*.bmp"))})
#
#     test_data = pd.DataFrame({'images': sorted(glob(os.path.join(test_path, "img") + "/*.bmp")),
#                               'masks': sorted(glob(os.path.join(test_path, "gt") + "/*.bmp"))})
#     train_dataset = CrackData(df=train_data, transforms=train_transforms)
#     val_dataset = CrackData(df=val_data, transforms=val_data)
#     test_dataset = CrackData(df=test_data, transforms=test_transforms,img_size= True)
#
#
#     print(len(train_dataset),len(test_dataset),len(val_dataset))
#
#     training_data_loader = DataLoader(dataset=train_dataset, num_workers=1, batch_size=2, shuffle=True)
#     test_data_loader = DataLoader(dataset=test_dataset, num_workers=1, batch_size=2, shuffle=False)
#     val_data_loader = DataLoader(dataset=val_dataset, num_workers=1, batch_size=1, shuffle=False)

    # for inputs, labels in training_data_loader:
    #     img1 = inputs[0,:,:,:].squeeze().cpu().data.numpy()
    #     gt1 = labels[0, :, :, :].squeeze().cpu().data.numpy()
    #     import matplotlib.pyplot as plt
    #
    #     plt.suptitle("Name:{0}".format(training_data_loader.dataset), fontsize=10, x=0.5, y=0.98)
    #     _,ax = plt.subplots(1,2)
    #     ax[0].imshow(img1,"gray")
    #     ax[1].imshow(gt1,"gray")
    #     plt.pause(2)
    #     plt.close()
    #     img2 = inputs[1, :, :, :].squeeze().cpu().data.numpy()
    #     gt2 = labels[1, :, :, :].squeeze().cpu().data.numpy()
    #     import matplotlib.pyplot as plt
    #     _, ax = plt.subplots(1, 2)
    #     ax[0].imshow(img2,"gray")
    #     ax[1].imshow(gt2,"gray")
    #     plt.pause(2)
    #     plt.close()

    #
    # for inputs, labels, sizes  in test_data_loader:   #test output img shape
    #     img1 = inputs[0,:,:,:].squeeze().cpu().data.numpy()
    #     gt1 = labels[0, :, :, :].squeeze().cpu().data.numpy()
    #     print(sizes, list(zip(sizes[0].numpy().tolist(), sizes[1].numpy().tolist())))
    #     (ori_width, ori_height) = list(zip(sizes[0].numpy().tolist(), sizes[1].numpy().tolist()))[0]
    #     import matplotlib.pyplot as plt
    #
    #     plt.suptitle("Name:{0}".format(training_data_loader.dataset), fontsize=10, x=0.5, y=0.98)
    #     _,ax = plt.subplots(1,2)
    #     ax[0].imshow(img1,"gray")
    #     ax[1].imshow(gt1,"gray")
    #     plt.pause(2)
    #     plt.close()


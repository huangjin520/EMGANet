import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import pandas as pd
import gc


class CrackData(Dataset):
    def __init__(self, df, transforms=None):
        self.data = df
     
        self.transform = transforms
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data['images'].iloc[idx]).convert('RGB')
        gt = Image.open(self.data['masks'].iloc[idx]).convert('L')
        
        sample = {'image': img, 'gt': gt}
    
        sample = self.transform(sample)
    
        return sample['image'], sample['gt']


      


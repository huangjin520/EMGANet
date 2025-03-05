import torch.utils.data
from util.parser import get_parser_with_args

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
# from util.AverageMeter import AverageMeter, RunningMetrics
from util.transforms import train_transforms,test_transforms 
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from glob import glob
import os
from util.common import result_visual
import cv2
from util.AverageMeter import RunningMetrics,ContourSimilarityCalculator
from util.dataset_jin import CrackData
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = 'EMGANet/med_seg/pt/EMGANet.pt'   # the path of the model127\111\297/216/217
model = torch.load(path,map_location={'cuda:0':'cuda:0'})


print('===> Loading datasets')

from util.dataset_jin import CrackData
test_path = "EMGANet/Dataset/BUSI_WHU/test" # the path of validation

test_data = pd.DataFrame({'images': sorted(glob(os.path.join(test_path, "img") + "/*.bmp")),
              'masks': sorted(glob(os.path.join(test_path, "mask") + "/*.bmp"))})
test_dataset = CrackData(df = test_data,transforms=test_transforms, img_size=True)
test_loader = DataLoader(dataset=test_dataset, num_workers=8, batch_size=8, shuffle=False)
print(len(test_data.images.values))

name=0
c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()

running_metrics =  RunningMetrics(2)
cc_s = ContourSimilarityCalculator(2)

test_epoch_iou=[]
with torch.no_grad():
    tbar = tqdm(test_loader)
    for i, results in enumerate(tbar):
        batch_img, labels, ws, hs, *_   = results
        batch_img = batch_img.float().to(dev)
        labels = labels.long().to(dev)
        # print(batch_img2 .size())
        
    # if torch.unique(labels).size()[0]==2 :
    # if torch.sum(labels==1)>512:
        cd_preds= model(batch_img)
        _, cd_preds = torch.max(cd_preds, 1)
        
        running_metrics.update(labels.data.cpu().numpy(),cd_preds.data.cpu().numpy())
        cc_s.update(cd_preds.data.cpu().numpy(),labels.data.cpu().numpy(),img_sizes=(ws, hs))
       
            
score = running_metrics.get_scores()
print(score )

all_df=cc_s.all_df
print(all_df)

average_values = all_df.mean()
print(average_values)
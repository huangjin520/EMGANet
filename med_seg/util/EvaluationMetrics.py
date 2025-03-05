# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np
eps=np.finfo(float).eps



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores, cls_iu, m_1 = cm2score(self.sum)
        scores.update(cls_iu)
        scores.update(m_1)
        return scores


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    acc_cls_ = tp / (sum_a1 + np.finfo(np.float32).eps)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*acc_cls_ * precision / (acc_cls_ + precision + np.finfo(np.float32).eps)
    # ---------------------------------------------------------------------- #
    # 2. Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    cls_iu = dict(zip(range(n_class), iu))



    return {'Overall_Acc': acc,
            'Mean_IoU': mean_iu}, cls_iu, \
           {
        'precision_1': precision[1],
        'recall_1': acc_cls_[1],
        'F1_1': F1[1],}


class RunningMetrics(object):
    def __init__(self, num_classes):
        """
        Computes and stores the Metric values from Confusion Matrix
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param num_classes: <int> number of classes
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def __fast_hist(self, label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth-----jin b,h,w
        :param label_pred: <np.array> prediction-----jin b,h,w
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < self.num_classes)
        hist = np.bincount(self.num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, label_gts, label_preds):
        """
        Compute Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gts: <np.ndarray> ground-truths, (batchsize, h, w)
        :param label_preds: <np.ndarray> predictions  (batchsize, num_class, h, w)
        :return:
        """
        for lt, lp in zip(label_gts, label_preds):
            self.confusion_matrix += self.__fast_hist(lt.flatten(), lp.flatten())

    def reset(self):
        """
        Reset Confusion Matrix
        :return:
        """
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_cm(self):
        return self.confusion_matrix
    @staticmethod
    def calculate_kappa(class_confusion_matrix):
        total_samples = np.sum(class_confusion_matrix)

        po = np.trace(class_confusion_matrix) / total_samples

        pe = np.sum(np.sum(class_confusion_matrix, axis=0) * np.sum(class_confusion_matrix, axis=1)) / (total_samples ** 2)

        kappa = (po - pe) / (1 - pe + np.finfo(np.float32).eps)
        return kappa, pe

    def calculate_kappas_for_each_class(self):
        
        kappas_classes = {}
        pe_classes = {}
        if self.num_classes == 2:
            #2分类返回一个值，多分类返回多个值
            kappa_i, pe_i = self.calculate_kappa(self.confusion_matrix)
            kappas_classes.update({"2分类1": kappa_i})
            pe_classes.update({"2分类1": pe_i})

            return kappas_classes, pe_classes
            
        else:
            for i in range(self.num_classes):
                # Extract values for the i-th class
                TP_i = self.confusion_matrix[i, i]
                FP_i = np.sum(self.confusion_matrix[:, i]) - TP_i
                FN_i = np.sum(self.confusion_matrix[i, :]) - TP_i
                TN_i = np.sum(self.confusion_matrix) - TP_i - FP_i - FN_i
                # Create a new confusion matrix for the i-th class
                class_confusion_matrix = np.array([[TP_i, FP_i], [FN_i, TN_i]])
                # Calculate Kappa for the i-th class
                kappa_i, pe_i = self.calculate_kappa(class_confusion_matrix)
                kappas_classes.update({str(i): kappa_i})
                pe_classes.update({str(i): pe_i})

                return kappas_classes, pe_classes
    
    def get_local_iou(self,label_gts,label_preds):
        """
        Returns score about:
            - mean IU

        """
        hist = self.__fast_hist(label_gts.flatten(), label_preds.flatten())
        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)
        sum_a0 = hist.sum(axis=0)

        iu = tp / (sum_a1 + sum_a0 - tp + np.finfo(np.float32).eps)
        mean_iu = np.nanmean(iu)

        each_iou={'local_IoU': mean_iu}
        each_iou.update(zip(range(self.num_classes), iu))

        return each_iou

    def get_scores(self):
        """
        Returns score about:
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :return:
        """
        hist = self.confusion_matrix
        tp = np.diag(hist)
        sum_a1 = hist.sum(axis=1)
        sum_a0 = hist.sum(axis=0)

        # ---------------------------------------------------------------------- #
        # 1. Accuracy & Class Accuracy
        # ---------------------------------------------------------------------- #
        acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

        # recall
        acc_cls_ = tp / (sum_a1 + np.finfo(np.float32).eps)

        # precision
        precision = tp / (sum_a0 + np.finfo(np.float32).eps)
        # ---------------------------------------------------------------------- #
        # 2. Mean IoU
        # ---------------------------------------------------------------------- #
        iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
        mean_iu = np.nanmean(iu)

        cls_iu = dict(zip(range(self.num_classes), iu))

        # F1 score
        F1 = 2 * acc_cls_ * precision / (acc_cls_ + precision + np.finfo(np.float32).eps)
        
        jaccard_per_class = tp / (sum_a1 + sum_a0 - tp + eps)   #Jaccard 等同于iou
        dice_per_class = 2 * tp / (sum_a1 + sum_a0 + eps)
        
        cls_jaccard = dict(zip(range(self.num_classes), jaccard_per_class))
        cls_dice = dict(zip(range(self.num_classes), dice_per_class))
        kappas_classes, pe_classes = self.calculate_kappas_for_each_class()

        # Average Jaccard Index and Dice Coefficient
        avg_jaccard = np.nanmean(jaccard_per_class)
        avg_dice = np.nanmean(dice_per_class)
        avg_F1 = np.nanmean(F1)
        avg_precision = np.nanmean(precision)
        avg_recall = np.nanmean(acc_cls_)
        

        scores = {'Overall_Acc': acc.round(4),
                'Mean_IoU': mean_iu.round(4),
                'Mean_Dice': avg_dice.round(4),
                'Mean_F1':avg_F1.round(4), 
                'Mean_precision':avg_precision.round(4), 
                'Mean_recall':avg_recall.round(4), 
                'Mean_Jaccard': avg_jaccard.round(4),
                }
        scores.update({"iou":cls_iu,
                       "jaccard":cls_jaccard,
                       "dice": cls_dice,
                       "precesion": precision,
                       "recall": acc_cls_,
                       "F1": F1,
                       "Kappa":kappas_classes, 
                       "Pe": pe_classes})

        scores.update({'precision_1': precision[1],
                       'recall_1': acc_cls_[1],
                       'F1_1': F1[1]})
        return scores

        

import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import directed_hausdorff
import pandas as pd
from medpy.metric.binary import assd as ASSD

class ContourSimilarityCalculator:
    """一次只能计算多个图片,支持batchsize"""
    def __init__(self, num_calsses) -> None:
        self.num_classes = num_calsses
        self.all_df = pd.DataFrame()
        
    def mask_gen(self, mask1, mask2):
        """mask1 对应的是pred,
           mask2 对应的是gt   jinjin"""
        self.mask1 = mask1
        self.mask2 = mask2
        self.mask1_gray = self.mask1.astype(np.uint8) * 255
        self.mask2_gray = self.mask2.astype(np.uint8) * 255
        self.contour1s, _ = cv2.findContours(self.mask1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contour2s, _ = cv2.findContours(self.mask2_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(self.contour1s)==1:
            # mask1 对应的是pred，这里有个筛选
            self.contour1s = self.contour1s
            
        if len(self.contour1s)>=2:
            """去除轮廓面小于200的"""
            if cv2.contourArea(max(self.contour1s, key=cv2.contourArea)) >200:
                self.contour1s = [contour for contour in self.contour1s if (cv2.contourArea(contour) != 0)&(cv2.contourArea(contour) >= 200)]
            else :
                self.contour1s = [(max(self.contour1s, key=cv2.contourArea))]
        
        if len(self.contour2s)==1:
            # mask1 对应的是pred，这里有个筛选
            self.contour2s = self.contour2s   
        if len(self.contour2s)>=2:
            """去除轮廓面小于200的"""
            if cv2.contourArea(max(self.contour2s, key=cv2.contourArea)) >200:
                self.contour2s = [contour for contour in self.contour2s if (cv2.contourArea(contour) != 0)&(cv2.contourArea(contour) >= 200)]
            else :
                self.contour2s = [(max(self.contour2s, key=cv2.contourArea))]


        """计算轮廓的周长"""
 
        self.contour1_perimeters = [cv2.arcLength(contour, True) for contour in self.contour1s]
        self.contour2_perimeters = [cv2.arcLength(contour, True) for contour in self.contour2s]

        self.contour1_perimeters_total = sum(self.contour1_perimeters)
        self.contour2_perimeters_total = sum(self.contour2_perimeters)
        # print(self.contour1_perimeters, self.contour2_perimeters)

        """计算面积"""
        # self.area1 = sum((self.mask1>0)).sum() #jinjinjin
        self.area1 = (self.mask1>0).sum()#####两个结果不一样，不明白
        
        self.area2 = (self.mask2>0).sum()
    

        self.contour1_perimeters_total, self.contour2_perimeters_total, self.area1, self.area2 = map(int, [self.contour1_perimeters_total, self.contour2_perimeters_total, self.area1, self.area2])

        return self.contour1_perimeters_total, self.contour2_perimeters_total, self.area1, self.area2

    def calculate_hu_moments(self, contours):
        # 计算所有轮廓的 Hu 不变矩
        hu_moments_list = []
        for contour in contours:
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments_list.append(hu_moments)
        return hu_moments_list
    
    def hu_moments_similarity(self):

        hu_moments_list1 = self.calculate_hu_moments(self.contour1s)
        hu_moments_list2 = self.calculate_hu_moments(self.contour2s)

        # 计算相似性矩阵
        similarity_matrix = np.zeros((len(hu_moments_list1), len(hu_moments_list2)))

        for i, hu_moments1 in enumerate(hu_moments_list1):
            for j, hu_moments2 in enumerate(hu_moments_list2):
                distance = euclidean(hu_moments1, hu_moments2)
                similarity = 1.0 / (1.0 + distance)  # 越小的距离越大的相似性
                similarity_matrix[i, j] = similarity

        return similarity_matrix

    def iou_similarity(self):
        # 计算交并比相似性
        intersection = np.logical_and(self.mask1, self.mask2).sum()
        union = np.logical_or(self.mask1, self.mask2).sum()
        iou = intersection / union
        return iou

    def dice_similarity(self):
        # 计算交并比相似性
        intersection = np.logical_and(self.mask1, self.mask2).sum()
        union =self.mask1.sum()+self.mask2.sum()
        dice = 2*intersection / union
        return dice

    def hd(self):
        # 豪斯多夫距离
        # gt_mask 和 pred_mask 是二值分割图像，可以是 numpy 数组或列表

        gt_coords = np.argwhere(self.mask1)
        pred_coords = np.argwhere(self.mask2)
        # 计算两个集合之间的豪斯多夫距离
        hausdorff_distance_gt_to_pred= directed_hausdorff(gt_coords, pred_coords)[0]
        hausdorff_distance_pred_to_gt= directed_hausdorff(pred_coords, gt_coords)[0]
        
        # 取两个方向的最大距离
        max_distance = np.array(max(hausdorff_distance_gt_to_pred, hausdorff_distance_pred_to_gt))
        
        return max_distance
    def hd95(self):

        gt_coords = np.argwhere(self.mask1)
        pred_coords = np.argwhere(self.mask2)
        # 计算两个集合之间的豪斯多夫距离
        hausdorff_distance_gt_to_pred = directed_hausdorff(gt_coords, pred_coords)[0]
        hausdorff_distance_pred_to_gt = directed_hausdorff(pred_coords, gt_coords)[0]
        hd95 = np.percentile(np.hstack((hausdorff_distance_gt_to_pred, hausdorff_distance_pred_to_gt)), 95)
        return hd95
    def assd(self):
        assd = ASSD(self.mask1, self.mask2)
        return assd


    def calculate_contour_euclidean_distance(self, contour1, contour2):
        return 0
        moments1 = cv2.moments(contour1)
        moments2 = cv2.moments(contour2)

        centroid1_x = int(moments1["m10"] / moments1["m00"])
        centroid1_y = int(moments1["m01"] / moments1["m00"])

        centroid2_x = int(moments2["m10"] / moments2["m00"])
        centroid2_y = int(moments2["m01"] / moments2["m00"])

        distance = np.sqrt((centroid2_x - centroid1_x) ** 2 + (centroid2_y - centroid1_y) ** 2)
        return distance

    def calculate_mask_contour_euclidean_distances(self):

        num_contours1 = len(self.contour1s)
        num_contours2 = len(self.contour2s)
        distance_matrix = np.zeros((num_contours1, num_contours2))

        for i in range(num_contours1):
            for j in range(num_contours2):
                distance = self.calculate_contour_euclidean_distance(self.contour1s[i], self.contour2s[j])
                distance_matrix[i, j] = distance

        return distance_matrix
    

    def calculate_contour_manhattan_distance(self, contour1, contour2):
        return 0
        moments1 = cv2.moments(contour1)
        moments2 = cv2.moments(contour2)

        centroid1_x = int(moments1["m10"] / moments1["m00"])
        centroid1_y = int(moments1["m01"] / moments1["m00"])

        centroid2_x = int(moments2["m10"] / moments2["m00"])
        centroid2_y = int(moments2["m01"] / moments2["m00"])

        distance = abs(centroid2_x - centroid1_x) + abs(centroid2_y - centroid1_y)
        return distance

    def calculate_mask_contour_manhattan_distances(self):

        num_contours1 = len(self.contour1s)
        num_contours2 = len(self.contour2s)
        distance_matrix = np.zeros((num_contours1, num_contours2))

        for i in range(num_contours1):
            for j in range(num_contours2):
                distance = self.calculate_contour_manhattan_distance(self.contour1s[i], self.contour2s[j])
                distance_matrix[i, j] = distance

        return distance_matrix
    def calculate_general_metrics(self, gt, pred): #典型二分类计算记结果jin
        labels_np = gt
        preds_np = pred
        c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
        tp = np.sum((labels_np == 1) & (preds_np == 1))
        tn = np.sum((labels_np == 0) & (preds_np == 0))
        fp = np.sum((labels_np == 0) & (preds_np == 1))
        fn = np.sum((labels_np == 1) & (preds_np == 0))
        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp
        tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
        epsilon = 1e-10
        P = tp / (tp + fp + epsilon)
        R = tp / (tp + fn + epsilon)
        F1 = 2 * P * R / (P + R + epsilon)
        
        IOU_0 = tn/(tn+fp+fn+ epsilon)
        IOU_1 = tp/(tp+fp+fn+ epsilon)
        mIOU = (IOU_0+IOU_1)/2
        OA = (tp+tn)/(tp+fp+tn+fn)
        p0 = OA
        pe = ((tp+fp)*(tp+fn)+(fp+tn)*(fn+tn))/(tp+fp+tn+fn+ epsilon)**2
        Kappa = (p0-pe)/(1-pe)
        return mIOU, OA, p0, pe, Kappa, P ,R ,  F1
    
    def one_image(self, pred, gt):
        """直接模型输出结果pred  ===torch.Size([ 256, 256])   放 _, cd_preds = torch.max(cd_preds, 1)
        gt ====torch.Size([256, 256])
        """
        one_image_dict = {}
        
        #jin 计算背景的iou 和dice ,其他几种参数对背景没有意义，
        mIOU, OA, p0, pe, Kappa, P ,R ,  F1 = self.calculate_general_metrics(gt, pred)
        keys_i = ["mIOU", "OA", "pe", "Kappa", "precise", "recall", "F1"]
        values_i = [mIOU, OA, pe, Kappa, P ,R ,  F1]
        one_image_dict.update({k: v for k, v in zip(keys_i, values_i)})
        
        mask1 = (pred==0)
        mask2 = (gt==0)
        self.mask_gen(mask1, mask2)
        iou_sim_0 = self.iou_similarity()
        dice_sim_0 = self.dice_similarity()
        one_image_dict.update({"0_iou": iou_sim_0})
        one_image_dict.update({"0_dice": dice_sim_0})
        
        #jin 计算目标类别的iou 和dice ,其他几种参数对背景没有意义，
        for i in range(1, self.num_classes):
            mask1 = (pred==i)
            mask2 = (gt==i)
            self.mask_gen(mask1, mask2)
            

            if (self.area1==0)|(self.area2==0):
                keys = [str(i) +"_"+ ii for ii in ["iou", "dice", "hu_moment", "euclidean", "manhattan", "hd", "hd95", "assd"]]
                values = [0, 0, 0, 0, 0, 0, 0, 0]
                one_image_dict.update({k: v for k, v in zip(keys, values)})
                return  one_image_dict

            
            hu_moments_sim_matrix = self.hu_moments_similarity()
            iou_sim = self.iou_similarity()
            dice_sim = self.dice_similarity()
            
            
            contour_euclidean_distances_matrix = self.calculate_mask_contour_euclidean_distances()
            contour_manhattan_distances_matrix = self.calculate_mask_contour_manhattan_distances()

            hd = self.hd()
            hd95 = self.hd95()
            assd = self.assd()
            
            iou = iou_sim.round(3)
            dice = dice_sim.round(3)
            hu_moment = np.max(hu_moments_sim_matrix).round(3)
            euclidean = np.min(contour_euclidean_distances_matrix).round(2)
            manhattan = np.min(contour_manhattan_distances_matrix).round(2)
            hd = hd.round(3)
            hd95 = hd95.round(3)
            assd = assd.round(3)
            keys = [str(i) +"_"+ ii for ii in ["iou", "dice", "hu_moment", "euclidean", "manhattan", "hd", "hd95", "assd"]]
            values = [iou, dice, hu_moment, euclidean, manhattan, hd, hd95, assd]
            one_image_dict.update({k: v for k, v in zip(keys, values)})

        return  one_image_dict
    
    def reset(self):
        self.all_df = pd.DataFrame()
    
    def update(self, preds, gts, img_sizes=False):
        """ jin 直接模型输出结果pred  ===torch.Size([n, 256, 256])   放 _, cd_preds = torch.max(cd_preds, 1)计算后的结果
        gt ====torch.Size([n, 256, 256]),  
        img_sizes=(ws, hs)
        """   

        for i , (pred_i , gt_i) in enumerate( zip(preds, gts)):
            if  img_sizes is not False :
                #jin 计算的是原始像素

                ori_w, ori_h  = int(img_sizes[0][i]), int(img_sizes[1][i])
                
                # pred_i , gt_i = np.resize(pred_i,(ori_h, ori_w,)), np.resize(gt_i, (ori_h, ori_w, ))
                pred_i , gt_i = cv2.resize(np.uint8(pred_i),( ori_w, ori_h)), cv2.resize(np.uint8(gt_i), ( ori_w, ori_h))
            
            #jin 计算的缩放后的像素
            one_image_dict = self.one_image(pred_i , gt_i)
            keys = ['C_pred', 'C_gt', 'S_pred', 'S_gt']
            values = [self.contour1_perimeters_total, self.contour2_perimeters_total, self.area1, self.area2]
            one_image_dict.update({k: v for k, v in zip(keys, values)})
            
            pd_i =  pd.DataFrame(one_image_dict, index=[0])
            self.all_df = pd.concat([self.all_df, pd_i]).fillna(0)
            
            
        

        

#%%
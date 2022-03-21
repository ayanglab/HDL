import copy
from copy import deepcopy
import pandas as pd
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import h5py
from PIL import Image
import torch
from scipy.ndimage import filters
import cv2
import torch.nn.functional as F
import scipy.io as scio

train_transform = transforms.Compose([

    # transforms.ColorJitter(brightness=0.7, contrast=0.8, saturation=0.8),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.RandomAffine(15, scale=(0.9, 1.1)),
    # transforms.Resize([128,128]),
    # transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
#
test_transform = transforms.Compose([
    # transforms.ColorJitter(brightness=0.7, contrast=0.8, saturation=0.8),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms.Normalize([0.5], [0.5])

])





class biDataset_WMSeg(Dataset):

    def __init__(self, idxLeft=3652,idxRight=5808,boxed=True):  # crop_size,

        self.imlist = pd.read_csv('data_info.csv',index_col=0)
        self.imlist.index = np.arange(self.imlist.shape[0])
        self.imlist = self.imlist[idxLeft:idxRight]
        self.imlist.index = np.arange(self.imlist.shape[0])

        self.boxed = boxed
        self.box = pd.read_csv('cropbox.csv')


    def __getitem__(self, idx):
        if idx < len(self.imlist):
            subID = self.imlist.loc[idx]['subID']
            tag = self.imlist.loc[idx]['tag']
            ####crop
            df_sub = self.box[self.box.subID.isin([subID])]
            if df_sub[df_sub.tag.isin([tag])].shape[0] == 0:
                print('here')
            xmin, ymin, xmax, ymax = df_sub[df_sub.tag.isin([tag])][['xmin', 'ymin', 'xmax', 'ymax']].iloc[0]
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            f = h5py.File("./dataset/image/%s"%subID, 'r')

            phaseX = f[tag]['PhaseX']
            phaseY = f[tag]['PhaseY']
            phaseZ = f[tag]['PhaseZ']

            if self.boxed:
                labelX = phaseX[self.imlist.loc[idx]['img1'],xmin:xmax,ymin:ymax]/ 4096
                labelY = phaseY[self.imlist.loc[idx]['img1'],xmin:xmax,ymin:ymax]/ 4096
                labelZ = phaseZ[self.imlist.loc[idx]['img1'],xmin:xmax,ymin:ymax]/ 4096
            else:
                labelX = phaseX[self.imlist.loc[idx]['img1']] / 4096
                labelY = phaseY[self.imlist.loc[idx]['img1']] / 4096
                labelZ = phaseZ[self.imlist.loc[idx]['img1']] / 4096

            label = np.concatenate([[labelX],[labelY],[labelZ]])
            label = (label-0.5)/0.5


            mag = f[tag]['Mag']
            mag = np.array(mag)/np.max(np.array(mag))
            if self.boxed:
                img0 = mag[self.imlist.loc[idx]['img0'], xmin:xmax, ymin:ymax]
                img1 = mag[self.imlist.loc[idx]['img1'], xmin:xmax, ymin:ymax]
                img2 = mag[self.imlist.loc[idx]['img2'], xmin:xmax, ymin:ymax]
            else:
                img0 = mag[self.imlist.loc[idx]['img0']]
                img1 = mag[self.imlist.loc[idx]['img1']]
                img2 = mag[self.imlist.loc[idx]['img2']]

            img = np.concatenate([[img0], [img1],[img2]])
            img = (img-0.5)/0.5



            #####segmentation
            f = h5py.File("./dataset/label/%s" % subID, 'r')
            # print(subID)
            mag = f[tag]['label']

            if self.boxed:
                img0_seg = mag[self.imlist.loc[idx]['img0'], xmin:xmax, ymin:ymax]
                img1_seg = mag[self.imlist.loc[idx]['img1'], xmin:xmax, ymin:ymax]
                img2_seg = mag[self.imlist.loc[idx]['img2'], xmin:xmax, ymin:ymax]
            else:
                img0_seg = mag[self.imlist.loc[idx]['img0']]
                img1_seg = mag[self.imlist.loc[idx]['img1']]
                img2_seg = mag[self.imlist.loc[idx]['img2']]

            img_seg = np.array([img0_seg,
                                img1_seg,
                                img2_seg])

            f = h5py.File("./dataset/padding/%s" % subID, 'r')
            weightmap2 = torch.Tensor(f[tag]['W2'])
            if self.boxed:
                weightmap2 = weightmap2[xmin:xmax,ymin:ymax]
            else:
                weightmap2 = weightmap2[xmin:xmax, ymin:ymax]

            noisemap = 1 - weightmap2

            return {'image':img,'phase':label,'seg':img_seg,'W2':weightmap2,'noise':noisemap}




    def __len__(self):
        return len(self.imlist)





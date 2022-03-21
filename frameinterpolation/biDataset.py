import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import torch


def feature_normalize(data):
    #data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # mu = np.mean(data)
    # std = np.std(data)
    mu = 0.5
    std = 0.5
    data = (data - mu) / std


    return torch.Tensor([data])



class biDataset(Dataset):

    def __init__(self, filename,
                 idxLeft=3652, idxRight=5808):  # crop_size,

        self.imlist = pd.read_csv(filename, index_col=0)
        self.imlist.index = np.arange(self.imlist.shape[0])
        self.imlist = self.imlist[idxLeft:idxRight]
        self.imlist.index = np.arange(self.imlist.shape[0])

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

            f = h5py.File("./dataset/image/%s" % subID, 'r')
            mag = f[tag]['Mag']
            mag = np.array(mag)/4096
            img0 = mag[self.imlist.loc[idx]['img0'],xmin:xmax,ymin:ymax]
            label = mag[self.imlist.loc[idx]['img1'],xmin:xmax,ymin:ymax]
            img2 = mag[self.imlist.loc[idx]['img2'],xmin:xmax,ymin:ymax]
            img0 = feature_normalize(img0)
            label = feature_normalize(label)
            img2 = feature_normalize(img2)
            img = torch.cat([img0, img2])

            f = h5py.File("./dataset/label/%s" % subID, 'r')
            mag = f[tag]['label']
            img0_seg = mag[self.imlist.loc[idx]['img0'],xmin:xmax,ymin:ymax]
            label_seg = mag[self.imlist.loc[idx]['img1'],xmin:xmax,ymin:ymax]
            img2_seg = mag[self.imlist.loc[idx]['img2'],xmin:xmax,ymin:ymax]
            img_seg = torch.cat([img0_seg, img2_seg])

            f = h5py.File("./dataset/padding/%s" % subID, 'r')
            weightmap1 = torch.Tensor(f[tag]['W1'])
            weightmap1 = weightmap1[xmin:xmax,ymin:ymax]
            weightmap2 = torch.Tensor(f[tag]['W2'])
            weightmap2 = weightmap2[xmin:xmax,ymin:ymax]

            timeGap = np.array([np.ones((256,256)) * self.imlist.loc[idx]['img0'] / 50,
                                np.ones((256,256)) * self.imlist.loc[idx]['img1'] / 50,
                                np.ones((256,256)) * self.imlist.loc[idx]['img2'] / 50])
            label = torch.cat([label, torch.Tensor([label_seg])])


            return {'image': img, 'label': label, 'seg':img_seg,'W1':weightmap1,'W2':weightmap2,'timeGap':timeGap}

    def __len__(self):
        return len(self.imlist)



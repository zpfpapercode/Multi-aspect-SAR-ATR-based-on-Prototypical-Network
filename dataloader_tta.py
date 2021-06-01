from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
# seed=10
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
# torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# np.random.seed(int(seed))
# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
#
# plt.ion()   # interactive mode

class ClipSubstractMean(object):
  def __init__(self, b=0, g=0, r=0):
    self.means = np.array((r, g, b))

  def __call__(self, sample):
    video_x,video_label=sample['video_x'],sample['video_label']
    new_video_x=video_x - self.means
    return {'video_x': new_video_x, 'video_label': video_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        video_x, video_label = sample['video_x'], sample['video_label']

        # swap color axis because
        # numpy image: batch_size x H x W x C
        # torch image: batch_size x C X H X W

        video_x=np.array(video_x)
        video_label = [int(video_label)]
        video_x_tensor=torch.from_numpy(video_x)
        video_x_tensor=video_x_tensor.type(torch.FloatTensor)

        return {'video_x':video_x_tensor,'video_label':torch.Tensor(video_label)}


class mv_sar_tta(Dataset):
    def __init__(self,info_list,root_dir,transform=None):
        """
        Args:
            info_list (string): Path to the info list file with annotations.
            root_dir (string): Directory with all the video frames.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.file=open(info_list)
        self.sar_frame=list(csv.reader(self.file))
        self.root_dir = root_dir
        self.transform = transform
        self.pic_name=list(map(lambda x: x[0], self.sar_frame))
        self.pic_label = tuple(map(lambda x: int(x[1]), self.sar_frame))
        self.x=list(map(self.get_single_video_x,self.pic_name))
        self.y=self.pic_label

    def __len__(self):
        return len(self.sar_frame)


    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x=self.transform(x)
        y=self.y[idx]
        return x,y
        # video_path=os.path.join(self.root_dir,self.sar_frame[idx][0])
        # video_label = int(self.sar_frame[idx][1])
        # video_x = self.get_single_video_x(video_path,4)
        # sample = {'video_x': video_x, 'video_label': video_label}
        # if self.transform:
        #     sample = self.transform(sample)
        # return sample['video_x'],sample['video_label']

    def get_single_video_x(self,video_path,n_pics=4):
        video_path = os.path.join(self.root_dir, video_path)
        video_x=np.zeros((3,n_pics,64,64,3))
        pic_tta=os.listdir(video_path)
        for j,name in enumerate(pic_tta):
            pic_path=os.path.join(video_path,name)
            pics=os.listdir(pic_path)
            for i in range(n_pics):
                image_path = os.path.join(pic_path, pics[i])
                tmp_image = io.imread(image_path)
                video_x[j,i, :, :, :] = tmp_image

        video_x=torch.from_numpy(video_x)
        return video_x


if __name__=='__main__':
    root_list='./data/'
    info_list='./data/test_tta.csv'
    # tf=transforms.Compose([ClipSubstractMean(),ToTensor()])
    my_sar_mv=mv_sar_tta(info_list,root_list,transform=None)
    dataloder=DataLoader(my_sar_mv,batch_size=32,shuffle=True,num_workers=8)
    for s in dataloder:
        # print(s['video_x'].size()self.y[idx])
        x=s[0]
        x=x[:,1,:,:,:]

        print(x.size())
        break
        # s['video_label']=torch.squeeze(s['video_label'])
        # print(s['video_label'])














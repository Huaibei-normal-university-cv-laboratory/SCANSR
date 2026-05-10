#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: dataloader_DIV2K_memory.py
# Created Date: Saturday April 4th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:10:38 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import os
import cv2
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms as T


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.__preload__()

    def __preload__(self):
        try:
            self.hr_t2, self.lr_t2, self.hr_pd = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.hr_t2, self.lr_t2, self.hr_pd = next(self.dataiter)

        with torch.cuda.stream(self.stream):
            self.hr_t2 = self.hr_t2.cuda(non_blocking=True)
            self.lr_t2 = self.lr_t2.cuda(non_blocking=True)
            self.hr_pd = self.hr_pd.cuda(non_blocking=True)
            # self.ref = self.ref.cuda(non_blocking=True)
            # self.hr = (self.hr / 255.0 - 0.5) * 2.0
            # self.lr = (self.lr / 255.0 - 0.5) * 2.0
            # self.ref = (self.ref / 255.0 - 0.5) * 2.0

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        hr_t2 = self.hr_t2
        lr_t2 = self.lr_t2
        hr_pd = self.hr_pd
        # ref = self.ref
        self.__preload__()
        return hr_t2, lr_t2, hr_pd

    def __len__(self):
        """Return the number of images."""
        return len(self.loader)


class IXI_train(data.Dataset):
    def __init__(self, ixi_root, scale, crop_size, **kwargs):
        super(IXI_train, self).__init__()
        # path = opt['dataroot_GT']
        # hr_pd_path = os.path.join(ixi_root, 'train_HR_PD')
        hr_t2_path = os.path.join(ixi_root, 'train_HR_T2')
        lr_t2_path = os.path.join(ixi_root, "train_LR_T2", "X%d" % scale)
        hr_pd_path = os.path.join(ixi_root, "train_HR_PD")
        # lr_t2_path = os.path.join(ixi_root, 'train_LR_T2')
        # lr_t2_path = os.path.join(lr_t2_path, 'X%s' % scale)
        # hr_pds = sorted(os.listdir(hr_pd_path))
        # hr_t2s = sorted(os.listdir(hr_t2_path))
        # lr_t2s = sorted(os.listdir(lr_t2_path))
        if kwargs['dataloader_num'] == 'None':
            # hr_pds = sorted(os.listdir(hr_pd_path))
            hr_t2s = sorted(os.listdir(hr_t2_path))
            lr_t2s = sorted(os.listdir(lr_t2_path))
            hr_pds = sorted(os.listdir(hr_pd_path))
        else:
            num = int(kwargs['dataloader_num'])
            # hr_pds = sorted(os.listdir(hr_pd_path))[:num]
            hr_t2s = sorted(os.listdir(hr_t2_path))[:num]
            lr_t2s = sorted(os.listdir(lr_t2_path))[:num]
            hr_pds = sorted(os.listdir(hr_pd_path))[:num]
        # self.hr_pds = [os.path.join(hr_pd_path, i) for i in hr_pds]
        self.hr_t2s = [os.path.join(hr_t2_path, i) for i in hr_t2s]
        self.lr_t2s = [os.path.join(lr_t2_path, i) for i in lr_t2s]
        self.hr_pds = [os.path.join(hr_pd_path, i) for i in hr_pds]
        self.crop_size = crop_size

        self.scale = scale
        assert  len(hr_t2s) == len(lr_t2s) == len(hr_pds), 'ref image number != hr image number'
        print('total train image number(lr_t2):%s' % len(lr_t2s))
        print('total train image number(hr_pd):%s' % len(hr_pds))

    def __len__(self):
        # if self.train:
        return len(self.hr_t2s)
        # else:
        #     return 2000  # val时只测试200张图片

    def __getitem__(self, idx):
        # GT_img_path = self.GT_paths[idx]
        # ref_GT_img_path = self.GT_paths[idx].replace('T2', 'PD')
        # ref_GT_img_path = self.GT_paths[idx].replace('T2', 'PD')
        # ref_GT_img_path = self.GT_paths[idx].replace('T2', 'PD_noalign')
        # read image file
        # ref = cv2.imread(self.hr_pds[idx], cv2.IMREAD_UNCHANGED)
        hr_t2 = cv2.imread(self.hr_t2s[idx], cv2.IMREAD_UNCHANGED)
        lr_t2 = cv2.imread(self.lr_t2s[idx], cv2.IMREAD_UNCHANGED)
        hr_pd = cv2.imread(self.hr_pds[idx], cv2.IMREAD_UNCHANGED)

        # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # ref = torch.tensor(ref).unsqueeze(0).float() / 255.
        hr_t2 = torch.tensor(hr_t2).unsqueeze(0).float() / 255.
        lr_t2 = torch.tensor(lr_t2).unsqueeze(0).float() / 255.
        hr_pd = torch.tensor(hr_pd).unsqueeze(0).float() / 255.

        # if self.train:
        # _, H, W = hr_t2.shape
        # rnd_h = random.randint(0, max(0, H // self.scale - self.crop_size))
        # rnd_w = random.randint(0, max(0, W // self.scale - self.crop_size))
        # rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale), int(rnd_w * self.scale)
        # lr_t2 = lr_t2[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
        # # hr_pd = hr_pd[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
        # # im2_LQ = im2_LQ[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
        # hr_t2 = hr_t2[:, rnd_h_HR:rnd_h_HR + self.crop_size * self.scale, rnd_w_HR:rnd_w_HR + self.crop_size * self.scale]
        # hr_pd = hr_pd[:, rnd_h_HR:rnd_h_HR + self.crop_size * self.scale, rnd_w_HR:rnd_w_HR + self.crop_size * self.scale]
        # # ref = ref[:, rnd_h_HR:rnd_h_HR + self.crop_size * self.scale, rnd_w_HR:rnd_w_HR + self.crop_size * self.scale]

        return hr_t2, lr_t2, hr_pd


def GetLoader(dataset_roots,
              batch_size=16,
              random_seed=1234,
              **kwargs
              ):
    """Build and return a data loader."""
    if not kwargs:
        a = "Input params error!"
        raise ValueError(print(a))
    colorJitterEnable = kwargs["color_jitter"]
    colorConfig = kwargs["color_config"]
    degradation = kwargs["degradation"]
    image_scale = kwargs["image_scale"]
    lr_patch_size = kwargs["lr_patch_size"]
    subffix = kwargs["subffix"]
    num_workers = kwargs["dataloader_workers"]
    div2k_root = dataset_roots
    dataset_enlarge = kwargs["dataset_enlarge"]

    content_dataset = IXI_train(dataset_roots, image_scale, lr_patch_size, **kwargs)

    content_data_loader = data.DataLoader(
        dataset=content_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    prefetcher = DataPrefetcher(content_data_loader)
    return prefetcher

if __name__ == '__main__':
    print("=")
    train_dataset = '/data2/cwj/datasets/ixi/'
    kwargs = {'dataloader_num': 'None',
              'lr_patch_size': 64,
              'degradation': 'bicubic',
              'image_scale': 4,
              'subffix': 'png',
              'dataloader_workers': 6,
              'dataset_enlarge': 64,
              'color_jitter': False,
              'color_config':
                  {'brightness': 0.02,
                   'contrast': 0.02,
                   'saturation': 0.02,
                   'hue': 0.02},
              'enable_reshuffle': False
              }
    dataloader = GetLoader(train_dataset, batch_size=1, random_seed=1234, **kwargs)
    print(dataloader)
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: eval_dataloader_memory.py
# Created Date: Thursday March 11th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 20th April 2023 9:27:26 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


import os
import cv2
import glob
import torch
from tqdm import tqdm


class EvalDataset:
    def __init__(self,
                 dataset_name,
                 data_root,
                 batch_size=1,
                 degradation="bicubic",
                 image_scale=4,
                 subffix='png',
                 dataloader_num='None'):
        """Initialize and preprocess the urban100 dataset."""
        self.data_root = data_root
        self.degradation = degradation
        self.image_scale = image_scale
        self.dataset_name = dataset_name
        self.subffix = subffix
        self.dataset = []
        self.pointer = 0
        self.batch_size = 1

        if self.dataset_name.lower() == "ixi":
            self.dataset_name = "ixi"
        elif self.dataset_name.lower() == "brats2018":
            self.dataset_name = "BraTs2018"
        elif self.dataset_name.lower() == "set14":
            self.dataset_name = "Set14"
        elif self.dataset_name.lower() == "b100":
            self.dataset_name = "B100"
        elif self.dataset_name.lower() == "urban100":
            self.dataset_name = "Urban100"
        else:
            raise FileNotFoundError
        print("%s dataset is used!" % self.dataset_name)
        self.dataloader_num = dataloader_num
        self.__preprocess__()
        self.num_images = len(self.dataset)

        # c_transforms  = []
        # c_transforms.append(T.ToTensor())
        # c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        # self.img_transform = T.Compose(c_transforms)


    def __preprocess__(self):
        """Preprocess the Urban100 dataset."""
        if self.dataset_name == 'ixi':
            hr_t2_path = os.path.join(self.data_root, "val_HR_T2")
            lr_t2_path = os.path.join(self.data_root, "val_LR_T2", "X%d" % self.image_scale)
            hr_pd_path = os.path.join(self.data_root, "val_HR_PD")
        # ref_path = os.path.join(self.data_root, "val_HR_PD")
        elif self.dataset_name == 'BraTs2018':
            hr_t2_path = os.path.join(self.data_root, "val_HR_T2")
            lr_t2_path = os.path.join(self.data_root, "val_LR_T2", "X%d" % self.image_scale)
            hr_pd_path = os.path.join(self.data_root, "val_HR_T1")

        print("Evaluation dataset HR path: %s" % hr_t2_path)
        print("Evaluation dataset LR path: %s" % lr_t2_path)
        print("Evaluation dataset LR path: %s" % hr_pd_path)
        assert os.path.exists(hr_t2_path)
        assert os.path.exists(lr_t2_path)
        assert os.path.exists(hr_pd_path)
        # assert os.path.exists(ref_path)
        if self.dataloader_num == 'None':
            hr_t2_files = sorted(os.listdir(hr_t2_path))
            lr_t2_files = sorted(os.listdir(lr_t2_path))
            hr_pd_files = sorted(os.listdir(hr_pd_path))
            # ref_files = sorted(os.listdir(ref_path))
        else:
            hr_t2_files = sorted(os.listdir(hr_t2_path))[:self.dataloader_num]
            lr_t2_files = sorted(os.listdir(lr_t2_path))[:self.dataloader_num]
            hr_pd_files = sorted(os.listdir(hr_pd_path))[:self.dataloader_num]
            # ref_files = sorted(os.listdir(ref_path))[:self.dataloader_num]

        hr_t2_file_paths = [os.path.join(hr_t2_path, i) for i in hr_t2_files]
        lr_t2_file_paths = [os.path.join(lr_t2_path, i) for i in lr_t2_files]
        hr_pd_file_paths = [os.path.join(hr_pd_path, i) for i in hr_pd_files]
        # ref_file_paths = [os.path.join(ref_path, i) for i in ref_files]

        print("processing %s images..." % self.dataset_name)


        for idx in tqdm(range(len(lr_t2_file_paths))):
            # ref = cv2.imread(ref_file_paths[idx], cv2.IMREAD_UNCHANGED)
            hr_t2 = cv2.imread(hr_t2_file_paths[idx], cv2.IMREAD_UNCHANGED)
            lr_t2 = cv2.imread(lr_t2_file_paths[idx], cv2.IMREAD_UNCHANGED)
            hr_pd = cv2.imread(hr_pd_file_paths[idx], cv2.IMREAD_UNCHANGED)
            # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            # ref = torch.tensor(ref).unsqueeze(0).float() / 255.
            hr_t2 = torch.tensor(hr_t2).unsqueeze(0).float() / 255.
            lr_t2 = torch.tensor(lr_t2).unsqueeze(0).float() / 255.
            hr_pd = torch.tensor(hr_pd).unsqueeze(0).float() / 255.
            self.dataset.append((hr_t2, lr_t2, hr_pd))

        print('Finished preprocessing the ixi Validation dataset, total image number: %d...' % len(self.dataset))

    def __call__(self):
        """Return one batch images."""
        if self.pointer >= self.num_images:
            self.pointer = 0
        hr_t2 = self.dataset[self.pointer][0]
        lr_t2 = self.dataset[self.pointer][1]
        hr_pd = self.dataset[self.pointer][2]
        # hr = (hr / 255.0 - 0.5) * 2.0
        # lr = (lr / 255.0 - 0.5) * 2.0
        hr_t2 = hr_t2.unsqueeze(0)
        lr_t2 = lr_t2.unsqueeze(0)
        hr_pd = hr_pd.unsqueeze(0)
        self.pointer += 1
        return hr_t2, lr_t2, hr_pd

    def __len__(self):
        return self.num_images

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.data_root + ')'

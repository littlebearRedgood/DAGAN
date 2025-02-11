# -*- coding: utf-8 -*-
# @Author   : Honghao Xu
# @Time     : 2024/3/1 15:09
import os
import cv2
import numpy as np
import torch


class DatasetHandler:

    def __init__(self, train_path='../DataSets/LSUI/', test_path='../DataSets/test/'):
        self.train_path_input = os.path.join(train_path, 'input')
        self.train_path_gt = os.path.join(train_path, 'GT')
        self.test_path_input = os.path.join(test_path, 'input')
        self.test_path_gt = os.path.join(test_path, 'GT')

    def pre_dataset(self, data_list, path):
        path_list = os.listdir(path)
        path_list = [item for item in path_list if item.endswith(('.jpg', '.png', '.jpeg'))]
        path_list.sort(key=lambda x: int(os.path.splitext(x)[0]))
        for item in path_list:
            img_path = os.path.join(path, item)
            img_i = cv2.imread(img_path)
            if img_i is None:
                print("ZhiYin: " + img_path)
                continue
            img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)
            img_i = cv2.resize(img_i, (256, 256))
            data_list.append(img_i)

        data_list = np.array(data_list)
        data_list = data_list.astype('float32')
        data_list /= 255.0
        return torch.from_numpy(data_list).permute(0, 3, 1, 2)

    def init_dataset(self):
        x_train = self.pre_dataset([], self.train_path_input)
        y_train = self.pre_dataset([], self.train_path_gt)
        x_test = self.pre_dataset([], self.test_path_input)
        y_test = self.pre_dataset([], self.test_path_gt)
        return x_train, y_train, x_test, y_test

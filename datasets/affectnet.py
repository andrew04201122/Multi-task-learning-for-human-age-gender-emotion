# Copyright 2020 MISLAB, NTHU

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image


class ImageRecord(object):
    def __init__(self, raw):
        self._data = raw

    @property
    def path(self):
        return self._data[0]
    
    @property
    def label(self):
        return self._data[1]


class GenericImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 fold=None,
                 n_class=8,
                 ):
        self.root = root
        self.train = train
        self.transform = transform
        self.fold = fold
        self.n_class = n_class

        if not os.path.exists(self.root):
            raise FileNotFoundError

        self.img_root = ""
        self.list_path = ""
        self.img_list = []
        self.lbl_counts = [0] * self.n_class

    def __len__(self):
        return len(self.img_list)   

    def __getitem__(self, idx):
        record = self.img_list[idx]
        img = Image.open(record.path)
        class_idx = record.label

        if self.transform is not None:
            img = self.transform(img)

        return img, class_idx

    def _parse_list(self):
        self._parse_class_names()

        annotations = np.load(self.list_path, allow_pickle=True)
        for row in annotations:
            fname = row[0]
            class_idx = int(row[-1])
            if class_idx < self.n_class:
                record = ImageRecord((os.path.join(self.img_root, fname), class_idx))
                self.img_list.append(record)

                self.lbl_counts[class_idx] += 1

    def _parse_class_names(self):
        self.class_dict_encode = {}
        self.class_dict_decode = {}

        class_desc_file = 'datasets/classInd.txt'
        class_info = pd.read_csv(class_desc_file, sep=' ', header=None)
        for _, row in class_info.iterrows():
            class_idx, class_name = row
            self.class_dict_decode[class_idx] = class_name
            self.class_dict_encode[class_name] = class_idx

    def encode_class(self, class_name):
        return self.class_dict_encode[class_name]

    def decode_class(self, class_idx):
        return self.class_dict_decode[class_idx]


class AffectNet(GenericImageDataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 fold=None,
                 n_class=8,
                 ):
        super().__init__(root, train, transform, fold, n_class)

        self.img_root = os.path.join(self.root, 'all_data')
        self.list_path = "data_training_all.npy" if self.train else "data_validation_final.npy"
        self.list_path = os.path.join(self.root, self.list_path)

        if self.n_class > 8:
            raise NotImplementedError('Maximum num of classes is 8')

        self.img_list = []
        self._parse_list()


class Collected_dataset(GenericImageDataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 fold=1,
                 n_class=7,
                 ):
        super().__init__(root, train, transform, fold, n_class)

        self.img_root = os.path.join(self.root, 'all_data')
        self.list_path = f"fold{self.fold}_training.npy" if self.train else f"fold{self.fold}_validation.npy"
        self.list_path = os.path.join(self.root, self.list_path)

        if self.n_class > 7:
            raise NotImplementedError('Maximum num of classes is 7')

        if self.fold is None:
            raise ValueError('Fold is required')

        self.img_list = []
        self._parse_list()


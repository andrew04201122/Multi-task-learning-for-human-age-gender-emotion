import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from class_labels import CLASS_LABEL_GENDER, CLASS_LABEL_AGE_ADIENCE, CLASS_LABEL_AGE_IGS

class ImageRecord(object):
    def __init__(self, raw):
        self._data = raw

    @property
    def path(self):
        return self._data[0]
    
    @property
    def gender(self):
        return self._data[1]

    @property
    def age(self):
        return self._data[2]


class Adience(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 train=True,
                 split=0,
                 age_igs=False,
                 transform=None,
                 aligned=False
                 ):
        self.root = root
        self.train = train
        self.split = split
        self.age_igs = age_igs
        self.aligned = aligned
        self.transform = transform

        if not os.path.exists(self.root):
            raise FileNotFoundError

        self.img_root = os.path.join(self.root, 'aligned')
        #print(self.img_root)
        if self.aligned:
            self.img_root = os.path.join(self.root, 'aligned_mtcnn')

        self.img_list = []
        self._parse_list()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        record = self.img_list[idx]

        img = Image.open(record.path)
        gender = record.gender
        age = record.age

        if self.transform is not None:
            img = self.transform(img)

        return img, gender, age

    def _parse_list(self):
        self._parse_class_names()
        self._parse_data_counts()

        folds = [0, 1, 2, 3, 4]
        val_fold = [folds.pop(self.split)]
        train_fold = folds

        fold_to_load = train_fold if self.train else val_fold

        for i in fold_to_load:
            infix = '_aligned' if self.aligned else ''
            if self.age_igs:
                annotation_path = os.path.join(self.root, 'labels', f'fold_{i}{infix}_igs.txt')
            else:
                annotation_path = os.path.join(self.root, 'labels', f'fold_{i}{infix}_adience.txt')
            
            annotation = pd.read_csv(annotation_path, delimiter='\t')
            for _, row in annotation.iterrows():
                user_id = row['user_id']
                fname = row['original_image']
                face_id = row['face_id']
                gender = row['gender']
                if gender not in ['m', 'f']:
                    continue
                gender = 0 if gender == 'm' else 1
                age = row['age']

                record = ImageRecord((os.path.join(self.img_root, user_id, f"landmark_aligned_face.{face_id}.{fname}"), gender, age))
                self.img_list.append(record)

    def _parse_class_names(self):
        self.gender_dict_encode = {}
        self.gender_dict_decode = {}

        for class_idx, class_name in CLASS_LABEL_GENDER.items():
            self.gender_dict_decode[class_idx] = class_name
            self.gender_dict_encode[class_name] = class_idx
        
        self.age_dict_encode = {}
        self.age_dict_decode = {}

        if self.age_igs:
            age_dict = CLASS_LABEL_AGE_IGS
        else:
            age_dict = CLASS_LABEL_AGE_ADIENCE

        for class_idx, class_name in age_dict.items():
            self.age_dict_decode[class_idx] = class_name
            self.age_dict_encode[class_name] = class_idx

    def _parse_data_counts(self):
        if self.aligned:
            annotation_path = os.path.join(self.root, 'labels', f'data_count_aligned.json')
        else:
            annotation_path = os.path.join(self.root, 'labels', f'data_count.json')
        
        with open(annotation_path, 'r') as f:
            data_counts = json.loads(f.read())
        
        age_label = "igs" if self.age_igs else "adience"
        age_counts = data_counts["age"][age_label]
        gender_counts = data_counts["gender"][age_label]

        gender_mapping = {0: 'm', 1: 'f'}

        age_count_all = np.array([age_counts["all"][str(i)] for i in range(len(self.age_dict_decode))], dtype=np.int)
        age_count_fold = np.array([age_counts[f"fold_{self.split}"][str(i)] for i in range(len(self.age_dict_decode))], dtype=np.int)
        gender_count_all = np.array([gender_counts["all"][gender_mapping[i]] for i in range(len(self.gender_dict_decode))], dtype=np.int)
        gender_count_fold = np.array([gender_counts[f"fold_{self.split}"][gender_mapping[i]] for i in range(len(self.gender_dict_decode))], dtype=np.int)

        if self.train:
            self.age_count = age_count_all - age_count_fold
            self.gender_count = gender_count_all - gender_count_fold
        else:
            self.age_count = age_count_fold
            self.gender_count = gender_count_fold


    def encode_gender(self, class_name):
        return self.gender_dict_encode[class_name]

    def decode_gender(self, class_idx):
        return self.gender_dict_decode[class_idx]

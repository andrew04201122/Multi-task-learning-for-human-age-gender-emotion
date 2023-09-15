import numpy as np
import cv2
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler
import config as cf
import torchvision
from adience import Adience
from torchvision.transforms import transforms
from utils_transform import misc
from utils_transform.transforms_factory import transforms_train, transforms_val
from utils_transform.transforms_affectnet import (RandomErasing, get_color_distortion,
                              get_gaussian_blur)
from datasets.affectnet import AffectNet, Collected_dataset
import torch
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
def getAffectNet():
    print('==================================================================')
    print('\nLoading emotion image datasets.....')
    
    data_root = R"C:\Users\User\Desktop\Adience_affectnet\affectnet"
    dset = AffectNet
    train_tfs = transforms_train()
    test_tfs = transforms_val()
    n_class = 8
    fold = None
    image_res = 224
    batch_size = 32
    n_workers = 0

    train_tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(image_res, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(),
        transforms.RandomApply([transforms.Lambda(get_gaussian_blur)], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        RandomErasing(probability=0.3, sh=0.4, r1=0.3)
    ])

    test_tfs = transforms.Compose([
        transforms.Resize(image_res),
        transforms.CenterCrop(image_res),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trainset = dset(root=data_root, train=True, transform=train_tfs, fold=fold, n_class=n_class)
    testset = dset(root=data_root, train=False, transform=test_tfs, fold=fold, n_class=n_class)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=cf.BATCH_SIZE, num_workers=n_workers, pin_memory=True, shuffle=True) 
    test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=cf.BATCH_SIZE, num_workers=n_workers)

    return train_loader, test_loader

def getAgeGenderImage():
    print('==================================================================')
    print('\nLoading age gender image datasets.....')
    
    data_root = "./Adience"

    train_tfs = transforms_train()
    test_tfs = transforms_val()

    dset = Adience
    split = 0
    age_igs = True  #True ==> igs, false ==> adience
    n_workers = 4

    trainset = dset(root=data_root, split=split, age_igs=age_igs, train=True, transform=train_tfs)
    testset = dset(root=data_root, split=split, age_igs=age_igs, train=False, transform=test_tfs)

    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=cf.BATCH_SIZE, num_workers=n_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=cf.BATCH_SIZE, num_workers=n_workers)

    return train_loader, test_loader


def draw_labels_and_boxes(img, boxes, labels, margin=0):
    for i in range(len(labels)):
        # get the bounding box coordinates
        left, top, right, bottom = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        width = right - left
        height = bottom - top
        img_h, img_w = img.shape[:2]

        x1 = max(int(left - margin * width), 0)
        y1 = max(int(top - margin * height), 0)
        x2 = min(int(right + margin * width), img_w - 1)
        y2 = min(int(bottom + margin * height), img_h - 1)

        # Color red
        color = (0, 0, 255)

        # classify label according to result
        age_label, emotion_label, gender_label = labels[i]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = '{} {} {}'.format(age_label, emotion_label, gender_label)
        cv2.putText(img, text, (left - 35, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return img


def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='path to the image')
    arg.add_argument('-v', '--video_path', help='path to the video file')
    arg.add_argument('-m', '--margin', help='margin around face', default=0.0)
    return arg.parse_args()


def crop_face(image, result):
    nb_detected_faces = len(result)

    cropped_face = np.empty((nb_detected_faces, cf.IMAGE_SIZE, cf.IMAGE_SIZE, 1))
    boxes = []
    # loop through detected face
    for i in range(nb_detected_faces):
        # coordinates of boxes
        bounding_box = result[i]['box']
        left, top = bounding_box[0], bounding_box[1]
        right, bottom = bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]

        # coordinates of cropped image
        x1_crop = max(int(left), 0)
        y1_crop = max(int(top), 0)
        x2_crop = int(right)
        y2_crop = int(bottom)

        face = image[y1_crop:y2_crop, x1_crop:x2_crop, :]
        face = cv2.resize(face, (cf.IMAGE_SIZE, cf.IMAGE_SIZE), cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.reshape(cf.IMAGE_SIZE, cf.IMAGE_SIZE, 1)

        cropped_face[i, :, :, :] = face
        boxes.append((x1_crop, y1_crop, x2_crop, y2_crop))

    return cropped_face, boxes

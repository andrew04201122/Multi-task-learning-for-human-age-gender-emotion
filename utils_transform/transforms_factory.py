import torch
from torchvision.transforms.functional import crop
from utils_transform.transforms import ToNumpy
from torchvision import transforms as tfs
from torchvision.transforms.transforms import Normalize, RandomErasing

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def transforms_train(
    precrop_size=256,
    crop_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.,
    color_jitter=0.4,
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    re_prob=0.,
    re_scale=(0.02, 0.33),
    re_ratio=(0.3, 3.3),
    re_value=0
):
    scale = tuple(scale or (0.8, 1.0))
    ratio = tuple(ratio or (3./4., 4./3.))
    primary_tfs = [
        tfs.Resize(precrop_size),
        tfs.RandomResizedCrop(crop_size, scale, ratio)
    ]
    if hflip > 0.:
        primary_tfs += [tfs.RandomHorizontalFlip(hflip)]
    if vflip > 0.:
        primary_tfs += [tfs.RandomVerticalFlip(vflip)]

    secondary_tfs = []
    if color_jitter > 0.:
        jitter_param = (float(color_jitter),) * 3
        secondary_tfs += [tfs.ColorJitter(*jitter_param)]

    final_tfs = []
    if use_prefetcher:
        final_tfs += [ToNumpy()]
    else:
        final_tfs += [
            tfs.ToTensor(),
            tfs.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ]
        if re_prob > 0.:
            final_tfs += [tfs.RandomErasing(re_prob, re_scale, re_ratio, re_value)]

    return tfs.Compose(primary_tfs + secondary_tfs + final_tfs)

def transforms_val(
    crop_size=224,
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD
):
    primary_tfs = [
        tfs.Resize(crop_size),
        tfs.CenterCrop(crop_size)
    ]
    
    final_tfs = []
    if use_prefetcher:
        final_tfs += [ToNumpy()]
    else:
        final_tfs += [
            tfs.ToTensor(),
            tfs.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ]

    return tfs.Compose(primary_tfs + final_tfs)

def transforms_val_fivecrop(
    precrop_size=256,
    crop_size=224,
    flip=False,
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD
):
    primary_tfs = [
        tfs.Resize(precrop_size),
    ]
    if flip:
        primary_tfs += [tfs.TenCrop(crop_size)]
    else:
        primary_tfs += [tfs.FiveCrop(crop_size)]
    
    final_tfs = []
    if use_prefetcher:
        final_tfs += [
            tfs.Lambda(lambda crops: torch.stack([ToNumpy()(crop) for crop in crops])),
        ]
    else:
        final_tfs += [
            tfs.Lambda(lambda crops: torch.stack([tfs.ToTensor()(crop) for crop in crops])),
            tfs.Lambda(lambda tensors: 
                torch.stack([tfs.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))(t) for t in tensors]))
        ]

    return tfs.Compose(primary_tfs + final_tfs)
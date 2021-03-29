import os
import tarfile
import torch.utils.data as data
import numpy as np
import torch


class gf1Segmentation(data.Dataset):

    def __init__(self, root, image_set='train', transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform

        self.image_set = image_set
        voc_root = self.root
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]
        assert (len(self.images) == len(self.masks))

    # just return the img and target in P[key]
    def __getitem__(self, index):
        # get the hyperspectral data and lables
        rsData = np.load(self.images[index])
        labelData = np.load(self.masks[index])

        img = torch.Tensor(rsData[:, :32, :32])
        target = torch.Tensor(labelData[:32, :32])

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)
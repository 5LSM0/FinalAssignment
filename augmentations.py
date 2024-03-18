import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F



def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)

        # Add an extra dimension to the mask so that it can be used as a target
        mask = mask.unsqueeze(0)
        return img, mask

class Resize(T.Resize):
    def __call__(self, img, mask):
        return (
            F.resize(img, self.size, self.interpolation),
            F.resize(mask, self.size, interpolation=Image.NEAREST),
        )

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, img, mask):
        if random.random() < self.flip_prob:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        img = pad_if_smaller(img, self.size)
        mask = pad_if_smaller(mask, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(img, (self.size, self.size))
        img = F.crop(img, *crop_params)
        mask = F.crop(mask, *crop_params)
        return img, mask

class RandomCropWithProbability(object):
    def __init__(self, size, probability, padding=None, pad_if_needed=False):
        self.size = size
        self.probability = probability
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    def __call__(self, img, mask):
        if random.random() < self.probability:
            img = pad_if_smaller(img, self.size)
            mask = pad_if_smaller(mask, self.size, fill=255)
            crop_params = T.RandomCrop.get_params(img, (self.size, self.size))
            img = F.crop(img, *crop_params)
            mask = F.crop(mask, *crop_params)
            return img, mask
        else:
            return img, mask

class RandomRotation(object):
    def __init__(self, degrees, resample=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.center = center

    def __call__(self, img, mask):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        img = F.rotate(img, angle, self.resample, expand=False, center=self.center)
        mask = F.rotate(mask, angle, Image.NEAREST, expand=False, center=self.center)
        return img, mask

class CenterCrop(T.Normalize):
    def __call__(self, img, mask):
        return super().__call__(img), super().__call__(mask)

class Normalize(T.Normalize):
    def __call__(self, img, mask):
        return super().__call__(img), mask

class ColorJitter(T.ColorJitter):
    def __call__(self, img, mask):
        return super().__call__(img), mask

class RandomResizedCrop(T.RandomResizedCrop):
    def __call__(self, img, mask):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return (
            F.resized_crop(img, i, j, h, w, self.size, self.interpolation),
            F.resized_crop(mask, i, j, h, w, self.size, Image.NEAREST),
        )

class ToTensor(object):
    def __call__(self, img, label):
        return F.to_tensor(img), torch.as_tensor(np.array(label, np.uint8, copy=True))


class DualTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, label):
        return self.transform(image), self.transform(label)


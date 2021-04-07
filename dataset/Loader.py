import numpy as np
import torch
import os
import argparse
import time

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import utils
import matplotlib.pyplot as plt


def get_simple_dataset_transform(image_dim):
    return transforms.Compose([
        transforms.Scale(image_dim),
        transforms.CenterCrop(image_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # Imagenet stats
    ])

# Image loader class
class Imageloader():
<<<<<<< HEAD
    def __init__(self, datadir, batch_size, newsize, shuffle = True):
        self.path = datadir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = newsize
    def loadimage(self):
        dataset_transform = transforms.Compose([
        transforms.Resize(self.size),           # scale shortest side to image_size
        transforms.CenterCrop(self.size),      # crop center image_size out
        transforms.ToTensor(),                # turn image from [0-255] to [0-1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.ImageFolder(self.path, dataset_transform)
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = self.shuffle)
        return train_loader
=======
    def __init__(self, datadir, batch_size, image_dim, shuffle=True):
        self.datadir = datadir
        self.batch_size = batch_size
        self.shuffle = shuffle
        # image is going to be rescaled to 
        self.image_dim = image_dim

    def loadimage(self):
        dataset_transform = get_simple_dataset_transform(self.batch_size)
        train_dataset = datasets.ImageFolder(self.datadir, dataset_transform)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return train_loader
>>>>>>> 563ea7e0d918b7a4405393780203ce7533770f7e

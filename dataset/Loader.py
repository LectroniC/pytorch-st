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

#imgae loader class
class Imageloader():
    def __init__(self, datadir, batch_size, shuffle = True):
        self.path = datadir
        self.batch_size = batch_size
        self.shuffle = shuffle
    def loadimage(self):
        dataset_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),           # scale shortest side to image_size
        transforms.CenterCrop(IMAGE_SIZE),      # crop center image_size out
        transforms.ToTensor(),                # turn image from [0-255] to [0-1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.ImageFolder(self.path, dataset_transform)
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = self.shuffle)
        return train_loader
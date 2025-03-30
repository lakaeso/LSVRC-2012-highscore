from torchvision.datasets import ImageNet, CIFAR100, ImageFolder

from torchvision import transforms

from torch.utils.data import DataLoader, RandomSampler

import torch

import torch.nn as nn

import pathlib

from torchvision.transforms.functional import pil_to_tensor

from torchvision.transforms import v2

from torch.utils.data.dataloader import default_collate

from utils import *

from model import ImageNetClassifier

import random

import numpy as np

# constants
PATH_TO_DATA = pathlib.Path('D:\\datasets\\IMAGENET2012')

NUM_EPOCH = 50

TRAIN_BATCH_SIZE = 256

TEST_BATCH_SIZE = 256

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42

# set seeds
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# transforms
transforms = v2.Compose([
    # todo: random flip and crop
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# datasets
dataset_train = ImageFolder(pathlib.Path('D:\\datasets\\IMAGENET2012\\train'), transform=transforms) # ImageNet(PATH_TO_DATA, "val", transform=transforms)
dataset_test = ImageFolder(pathlib.Path('D:\\datasets\\IMAGENET2012\\val'), transform=transforms) # ImageNet(PATH_TO_DATA, "val", transform=transforms)

# dataloader
dataloader_train = DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=collate_fn, persistent_workers=True)
dataloader_test = DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn, persistent_workers=True)

# model
model = ImageNetClassifier(DEVICE).to(DEVICE)

# load
model.load_state_dict(torch.load('./saved_models/baseline3.pt', weights_only=True))

# criterion and optim
criterion = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(model.parameters(), lr=1e-6)

# train loop
if __name__ == '__main__':
    for i_epoch in range(NUM_EPOCH):
        model.epoch_train(i_epoch, dataloader_train, dataloader_test, optim, criterion)
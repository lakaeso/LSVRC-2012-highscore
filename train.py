from torchvision.datasets import ImageNet, CIFAR100, ImageFolder

from torchvision import transforms

from torch.utils.data import DataLoader

import torch

import torch.nn as nn

import pathlib

from torchvision.transforms.functional import pil_to_tensor

from torchvision.transforms import v2

from torch.utils.data.dataloader import default_collate

from utils import *

from model import ImageNetClassifier


# constants
PATH_TO_DATA = pathlib.Path('D:\\datasets\\IMAGENET2012')

NUM_EPOCH = 30

TRAIN_BATCH_SIZE = 64

TEST_BATCH_SIZE = 64

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transforms
transforms = v2.Compose([
    # v2.Resize((500, 500)),,
    #v2.ToImage(),
    #v2.ToDtype(torch.float, scale=False),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# datasets
dataset_train = ImageFolder(pathlib.Path('D:\\datasets\\IMAGENET2012\\train'), transform=transforms) # ImageNet(PATH_TO_DATA, "val", transform=transforms)
dataset_test = ImageFolder(pathlib.Path('D:\\datasets\\IMAGENET2012\\val'), transform=transforms) # ImageNet(PATH_TO_DATA, "val", transform=transforms)

# dataloader
dataloader_train = DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=collate_fn, persistent_workers=True)
dataloader_test = DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=1, collate_fn=collate_fn, persistent_workers=True)

# model
model = ImageNetClassifier(DEVICE).to(DEVICE)

# criterion and optim
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())

# train loop
if __name__ == '__main__':
    for i_epoch in range(NUM_EPOCH):
        model.epoch_train(i_epoch, dataloader_train, dataloader_test, optim, criterion)
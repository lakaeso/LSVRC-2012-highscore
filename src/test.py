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

import random

import numpy as np

# constants
PATH_TO_DATA = pathlib.Path('D:\\datasets\\IMAGENET2012')

NUM_EPOCH = 500

TRAIN_BATCH_SIZE = 256

TEST_BATCH_SIZE = 512

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42

# set seeds
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# transforms
transforms = v2.Compose([
    # v2.Resize((500, 500)),,
    #v2.ToImage(),
    #v2.ToDtype(torch.float, scale=False),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# datasets
dataset_train = ImageFolder(pathlib.Path('D:\\datasets\\IMAGENET2012\\train'), transform=transforms) # ImageNet(PATH_TO_DATA, "val", transform=transforms)
dataset_test = ImageFolder(pathlib.Path('D:\\datasets\\IMAGENET2012\\val'), transform=transforms) # ImageNet(PATH_TO_DATA, "val", transform=transforms)

# dataloader
# dataloader_train = DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn, persistent_workers=True)
dataloader_test = DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn, persistent_workers=True)

# model
model = ImageNetClassifier(DEVICE).to(DEVICE)

# criterion and optim
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters())

# train loop
if __name__ == '__main__':
    model.load_state_dict(torch.load('./saved_models/baseline3.pt', weights_only=True), strict=False)
    model.eval()   
    print(model.get_competition_error(dataloader_test))
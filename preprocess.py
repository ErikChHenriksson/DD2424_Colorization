from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from skimage import color
import numpy as np
from PIL import Image
from torch import split
from functions import ab_to_quantization

DATA_DIR = "./data"


def get_orig_data(data):
    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    if data == 'cifar':
        training_data = datasets.CIFAR10(DATA_DIR, train=True, download=False,
                                         transform=transform_train)
        test_data = datasets.CIFAR10(DATA_DIR, train=False, download=False,
                                     transform=transform_test)
    elif data == 'imgnet':
        training_data = datasets.ImageNet(
            './ImageNet_dog', train=True, transform=transform_train)
        test_data = datasets.ImageNet(
            './ImageNet_dog', train=False, transform=transform_test)

    return training_data, test_data


def get_lab_data(train_rgb, test_rgb):
    # training_rgbs = []
    training_labs = []
    # test_rgbs = []
    test_labs = []
    training_data, test_data = train_rgb, test_rgb

    for img, _label in training_data:
        # training_rgbs.append(img)
        lab = color.rgb2lab(img.T).T
        training_labs.append(lab)

    for img, _label in test_data:
        # test_rgbs.append(img)
        lab = color.rgb2lab(img.T).T
        test_labs.append(lab)

    return training_labs, test_labs


class LabTrainingDataset(Dataset):
    def __init__(self, training_labs):
        self._training_labs = training_labs

    def __len__(self):
        return len(self._training_labs)

    def __getitem__(self, index):
        return self._training_labs[index]


class LabTestDataset(Dataset):
    def __init__(self, test_labs):
        self._test_labs = test_labs

    def __len__(self):
        return len(self._test_labs)

    def __getitem__(self, index):
        return self._test_labs[index]

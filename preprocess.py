from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from skimage import color
import numpy as np
from PIL import Image
from torch import split
from functions import *
from tqdm import tqdm

data_dir = "./data"


def get_orig_data(data):
    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    if data == 'cifar':
        training_data = datasets.CIFAR10(data_dir, train=True, download=False,
                                         transform=transform_train)
        test_data = datasets.CIFAR10(data_dir, train=False, download=False,
                                     transform=transform_test)
    elif data == 'imgnet':
        training_data = datasets.ImageNet(
            './ImageNet_dog', train=True, transform=transform_train)
        test_data = datasets.ImageNet(
            './ImageNet_dog', train=False, transform=transform_test)

    return training_data, test_data


def get_lab_data_train(train_rgb):
    training_labs = []
    q_list = np.load('./quantized_space/q_list.npy')

    for img, _label in tqdm(train_rgb):
        # training_rgbs.append(img)
        lab = color.rgb2lab(img.T)
        l = lab[:, :, 0]
        ab = lab[:, :, 1:]
        one_hot = one_hot_q(ab, q_list)
        training_labs.append([l, one_hot])

    return training_labs


def get_lab_data_test(test_rgb):
    test_labs = []
    q_list = np.load('./quantized_space/q_list.npy')

    for img, _label in test_rgb:
        # test_rgbs.append(img)
        lab = color.rgb2lab(img.T)
        l = lab[:, :, 0]
        ab = lab[:, :, 1:]
        one_hot = one_hot_q(ab, q_list)
        test_labs.append([l, one_hot])

    return test_labs


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

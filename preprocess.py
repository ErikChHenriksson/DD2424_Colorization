from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
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
    for img, _label in tqdm(train_rgb):
        # training_rgbs.append(img)
        lab = color.rgb2lab(img.T)
        l = lab[:, :, 0]
        ab = lab[:, :, 1:]
        one_hot = one_hot_q(ab, load_data=True)
        training_labs.append([l, one_hot])

    return training_labs


def get_lab_data_test(test_rgb):
    test_labs = []

    for img, _label in test_rgb:
        # test_rgbs.append(img)
        lab = color.rgb2lab(img.T)
        l = lab[:, :, 0]
        ab = lab[:, :, 1:]
        one_hot = one_hot_q(ab, load_data=True)
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


if __name__ == '__main__':
    """ This main is used to create and save data from a downloaded dataset """
    data = 'cifar'
    train_rgb, test_rgb = get_orig_data(data)
    training_labs = get_lab_data_train(train_rgb)
    test_labs = get_lab_data_test(test_rgb)
    training_dataset = LabTrainingDataset(training_labs)
    test_dataset = LabTestDataset(test_labs)

    num_train_samples = 5000     # Create mini subset of data set
    train_subset = Subset(training_dataset, np.arange(num_train_samples))
    test_subset = Subset(test_dataset, np.arange(num_train_samples))
    train_sampler = RandomSampler(train_subset)
    test_sampler = RandomSampler(test_subset)
    lab_training_loader = DataLoader(
        train_subset, sampler=train_sampler, batch_size=100, shuffle=True, num_workers=2)
    lab_test_loader = DataLoader(
        test_sampler, sampler=train_sampler, batch_size=100, shuffle=True, num_workers=2)
    """ lab_training_loader = DataLoader(training_dataset, batch_size=100,
                                     shuffle=True, num_workers=2)
    lab_test_loader = DataLoader(test_dataset, batch_size=100,
                                 shuffle=True, num_workers=2) """

    torch.save(lab_training_loader, 'dataloaders/' +
               data+'_lab_training_loader_mini.pth')
    torch.save(lab_test_loader, 'dataloaders/' +
               data+'_lab_test_loader_mini.pth')

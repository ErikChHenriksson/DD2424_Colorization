from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from skimage import color
import numpy as np
from PIL import Image
import torch

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
        training_data = datasets.ImageNet(DATA_DIR, train=True, download=False,
                                          transform=transform_train)
        test_data = datasets.ImageNet(DATA_DIR, train=False, download=False,
                                      transform=transform_test)

    return training_data, test_data


def get_lab_data(train_rgb, test_rgb):
    # training_rgbs = []
    training_labs = []
    # test_rgbs = []
    test_labs = []
    training_data, test_data = train_rgb, test_rgb

    for img, _label in training_data:
        # training_rgbs.append(img)
        training_labs.append(color.rgb2lab(img.T).T)

    for img, _label in test_data:
        # test_rgbs.append(img)
        test_labs.append(color.rgb2lab(img.T).T)

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


def load_data(data='cifar'):
    train_rgb, test_rgb = get_orig_data(data)

    training_labs, test_labs = get_lab_data(train_rgb, test_rgb)

    training_dataset = LabTrainingDataset(training_labs)
    test_dataset = LabTestDataset(test_labs)

    lab_training_loader = DataLoader(training_dataset, batch_size=100,
                                     shuffle=True, num_workers=2)
    lab_test_loader = DataLoader(test_dataset, batch_size=100,
                                 shuffle=True, num_workers=2)

    torch.save(lab_training_loader, 'dataloaders/' +
               data+'_lab_training_loader.pth')
    torch.save(lab_training_loader, 'dataloaders/'+data+'_lab_test_loader.pth')

    return lab_training_loader, lab_test_loader


if __name__ == '__main__':
    """ Running this main function will save the dataloaders to dataloader/{dataloader_name} """
    load_data('imgnet')

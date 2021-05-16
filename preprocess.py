from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from skimage import color
import numpy as np
from PIL import Image

DATA_DIR = "./data"


def get_orig_data():
    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    training_data = datasets.CIFAR10(DATA_DIR, train=True, download=False,
                                     transform=transform_train)
    test_data = datasets.CIFAR10(DATA_DIR, train=False, download=False,
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


def load_data():
    train_rgb, test_rgb = get_orig_data()

    training_labs, test_labs = get_lab_data(train_rgb, test_rgb)

    training_dataset = LabTrainingDataset(training_labs)
    test_dataset = LabTestDataset(test_labs)

    lab_training_loader = DataLoader(training_dataset, batch_size=100,
                                     shuffle=True, num_workers=2)
    lab_test_loader = DataLoader(test_dataset, batch_size=100,
                                 shuffle=True, num_workers=2)

    return lab_training_loader, lab_test_loader


if __name__ == '__main__':
    load_data()

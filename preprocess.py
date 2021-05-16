from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from skimage import color
import numpy as np

DATA_DIR = "./data"


def get_orig_data():
    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    training_data = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                     transform=transform_train)
    test_data = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                 transform=transform_test)

    return training_data, test_data


def get_lab_data():
    # training_rgbs = []
    training_labs = []
    # test_rgbs = []
    test_labs = []
    training_data, test_data = get_orig_data()

    for img, _label in training_data:
        # training_rgbs.append(img)
        training_labs.append(color.rgb2lab(img.T).T)

    for img, _label in test_data:
        # test_rgbs.append(img)
        test_labs.append(color.rgb2lab(img.T).T)

    return training_labs, test_labs


training_labs, test_labs = get_lab_data()


class LabTrainingDataset(Dataset):
    def __init__(self):
        self._training_labs = training_labs

    def __len__(self):
        return len(self._training_labs)

    def __getitem__(self, index):
        return self._training_labs[index]


class LabTestDataset(Dataset):
    def __init__(self):
        self._test_labs = test_labs

    def __len__(self):
        return len(self._test_labs)

    def __getitem__(self, index):
        return self._test_labs[index]


training_dataset = LabTrainingDataset()
test_dataset = LabTestDataset()

lab_training_loader = DataLoader(training_dataset, batch_size=100,
                                 shuffle=True, num_workers=2)
lab_test_loader = DataLoader(test_dataset, batch_size=100,
                             shuffle=True, num_workers=2)



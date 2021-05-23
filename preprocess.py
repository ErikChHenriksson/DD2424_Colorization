import pickle
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
        training_labs.append([l, one_hot.T])

    return training_labs


def get_lab_data_test(test_rgb):
    test_labs = []

    for img, _label in test_rgb:
        # test_rgbs.append(img)
        lab = color.rgb2lab(img.T)
        l = lab[:, :, 0]
        ab = lab[:, :, 1:]
        one_hot = one_hot_q(ab, load_data=True)
        test_labs.append([l, one_hot.T])

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


def create_mini_set():
    data = 'cifar'
    train_rgb, test_rgb = get_orig_data(data)
    num_train_samples = 100  # Create mini subset of data set
    train_subset = Subset(train_rgb, range(0, num_train_samples))
    # test_subset = Subset(test_rgb, range(0, num_train_samples))
    training_labs = get_lab_data_train(train_subset)
    # test_labs = get_lab_data_test(test_subset)
    training_dataset = LabTrainingDataset(training_labs)
    # test_dataset = LabTestDataset(test_labs)
    lab_training_loader = DataLoader(training_dataset, batch_size=10,
                                     shuffle=True, num_workers=2)
    # lab_test_loader = DataLoader(test_dataset, batch_size=100,
    #                              shuffle=True, num_workers=2)

    torch.save(lab_training_loader, 'dataloaders/' +
               data+'_lab_training_loader_subset'+str(num_train_samples)+'.pth')
    # torch.save(lab_test_loader, 'dataloaders/' +
    #            data+'_lab_test_loader_mini.pth')


def save_np_data(data_rgb, size, q=246):
    num_samp = len(data_rgb)
    _, h, w = data_rgb[0][0].shape

    input_data = []
    target_data = []

    for i in tqdm(range(size)):
        img = data_rgb[i][0]
        lab = color.rgb2lab(img.T)
        l = lab[:, :, 0]
        ab = lab[:, :, 1:]
        one_hot = one_hot_q(ab, load_data=True)

        l = np.array(l).T
        one_hot = np.array(one_hot).T
        input_data.append(l)
        target_data.append(one_hot)

    np.save(f'./data_np/train_X{size}.npy', input_data)
    np.save(f'./data_np/train_y{size}.npy', target_data)
    return


if __name__ == '__main__':
    data = 'cifar'
    train_rgb, test_rgb = get_orig_data(data)
    save_np_data(train_rgb, 10000)

    # create_mini_set()

    # input_data = np.load('./data_np/train_data.npy', allow_pickle=True)
    # print(input_data[0][0].shape)
    # print(input_data[0][1].shape)

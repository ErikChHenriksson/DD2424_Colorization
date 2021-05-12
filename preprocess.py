import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np


def download_data():
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train = datasets.CIFAR10("./data", train=True, download=False,
                             transform=transform_train)

    test = datasets.CIFAR10("./data", train=False, download=False,
                            transform=transform_test)

    trainset = torch.utils.data.DataLoader(
        train, batch_size=100, shuffle=False, num_workers=2)
    testset = torch.utils.data.DataLoader(
        test, batch_size=100, shuffle=True, num_workers=2)

    return [trainset, testset]


def normalize(X):
    max = np.max(X, axis=0)
    min = np.min(X, axis=0)
    norm_data = (X-min) / (max-min)

    return norm_data


def preprocess():
    # trainset, testset = download_data()

    # trainset_lab = []
    # for batch in trainset:
    #     for image in batch[0]:
    #         lab = color.rgb2lab(image.T)
    #         trainset_lab.append(lab)

    # torch.save(trainset_lab, './processed_data/trainset_lab.pt')

    trainset_lab = torch.load('./processed_data/trainset_lab.pt')
    # trainset_lab_norm = normalize(trainset_lab)

    print(trainset_lab[0])

    # mean = np.mean(np.mean(trainset_lab, axis=0))
    # std = np.std(np.std(trainset_lab, axis=0))

    # mean = np.mean(trainset_lab, axis=1, keepdims=True)
    # std_dev = np.std(trainset_lab, axis=1, keepdims=True)
    # norm_trainset_lab = (trainset_lab - mean) / std_dev

    # x = trainset_lab[1]

    # plt.imshow(x[:, :, 0].T, cmap='gray')
    # plt.show()


if __name__ == '__main__':
    preprocess()

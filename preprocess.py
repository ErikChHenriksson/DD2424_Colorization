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


def getQ(pixels):
    colors = np.zeros((22, 22))

    for p in pixels:
        a, b = p
        colors[get_index(a), get_index(b)] = 1

    return np.count_nonzero(colors)


def get_index(num):
    (num + 110) / 10


def preprocess():
    # trainset, testset = download_data()

    # trainset_lab = []
    # for batch in trainset:
    #     for image in batch[0]:
    #         lab = color.rgb2lab(image.T)
    #         trainset_lab.append(lab)

    # torch.save(trainset_lab, './processed_data/trainset_lab.pt')

    # trainset_lab = torch.load('./processed_data/trainset_lab.pt')
    # # trainset_lab_norm = normalize(trainset_lab)

    # ab_pairs0 = []

    # count = 0
    # for img in trainset_lab[:5000]:
    #     for x in img[:, :, 1:]:
    #         for y in x:
    #             ab_pairs0.append(y)
    #     count += 1
    #     print(count)

    # print(len(ab_pairs0))
    # # print(ab_pairs[0])

    # torch.save(ab_pairs0, './processed_data/ab_values5000.pt')

    ab_values = torch.load('./processed_data/ab_values5000.pt')

    print(len(ab_values))

    # plt.scatter(a_vals, b_vals)
    # plt.show()

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

import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np


def download_data():
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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


def preprocess():
    trainset, testset = download_data()

    trainset_lab = []
    for batch in trainset:
        trainset_lab.append(color.rgb2lab(batch[0].reshape(100, 32, 32, 3)))
        break

    # x, y = batch[0][0], batch[1][0]

    # https://discuss.pytorch.org/t/is-there-rgb2lab-function-for-pytorch-data-varaible-or-tensor-or-torch-read-from-an-image/15594

    # ndarray = x.to('cpu').numpy()

    x = trainset_lab[0][0]

    plt.imshow(x, cmap='gray')
    plt.show()


if __name__ == '__main__':
    preprocess()

""" from skimage import io, color
rgb = io.imread(filename)
lab = color.rgb2lab(rgb)
 """

'''
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    trainset = torchvision.datasets.CIFAR10(
       root='./data', train=True, download=True, transform=transform_train)
   trainloader = torch.utils.data.DataLoader(
       trainset, batch_size=128, shuffle=True, num_workers=2)

   testset = torchvision.datasets.CIFAR10(
       root='./data', train=False, download=True, transform=transform_test)
   testloader = torch.utils.data.DataLoader(
       testset, batch_size=100, shuffle=False, num_workers=2)
    '''

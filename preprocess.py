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
        train, batch_size=100, shuffle=True, num_workers=2)
    testset = torch.utils.data.DataLoader(
        test, batch_size=100, shuffle=True, num_workers=2)

    return [trainset, testset]


def normalize(X):
    max = np.max(X, axis=0)
    min = np.min(X, axis=0)
    norm_data = (X-min) / (max-min)

    return norm_data


def get_index(num):
    (num + 110) / 10


def getQ(pixels):
    colors = np.zeros((22, 22))

    for p in pixels:
        # (a, b) = p
        colors[get_index(p[0]), get_index(p[1])] = 1

    return np.count_nonzero(colors)


def preprocess_and_save():
    trainset, testset = download_data()
    labX = []
    laby = []
    for batch in trainset:
        for image in batch[0]:
            lab = color.rgb2lab(image.T)
            L = lab[:, :, 0]
            ab = lab[:, :, 1:]
            labX.append(L)
            laby.append(ab)

    # torch.save(trainset_lab, './processed_data/trainset_lab.pt')

    labX_norm = normalize(labX)

    labXy = np.array([np.array(labX_norm), np.array(laby)])

    print(labXy.shape)

    # torch.save(labXy, './processed_data/trainset_lab_norm.pt')

    # ab_pairs0 = []
    # count = 0
    # for img in trainset_lab[:5000]:
    #     for x in img[:, :, 1:]:
    #         for y in x:
    #             ab_pairs0.append(y)
    #     count += 1
    #     print(count)

    # torch.save(ab_pairs0, './processed_data/ab_values5000.pt')


def load_data():
    # trainset_lab = torch.load('./processed_data/trainset_lab.pt')
    ab_values = torch.load('./processed_data/ab_values5000.pt')

    # print(np.array(ab_values)[:, 0])

    # q = getQ(ab_values)

    # print(q)

    # plt.scatter(np.array(ab_values)[:, 0], np.array(ab_values)[:, 1])
    # plt.show()

    # mean = np.mean(np.mean(trainset_lab, axis=0))
    # std = np.std(np.std(trainset_lab, axis=0))

    # mean = np.mean(trainset_lab, axis=1, keepdims=True)
    # std_dev = np.std(trainset_lab, axis=1, keepdims=True)
    # norm_trainset_lab = (trainset_lab - mean) / std_dev

    # x = trainset_lab[1]

    # plt.imshow(x[:, :, 0].T, cmap='gray')
    # plt.show()

# def resize_img(img, HW=(256,256), resample=3):
# 	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))


if __name__ == '__main__':
    preprocess_and_save()
    # load_data()

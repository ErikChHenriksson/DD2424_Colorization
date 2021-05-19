from model_cifar import Colorizer
from preprocess import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch import split
import torch
from tqdm import tqdm
from functions import find_k_nearest_q, one_hot_quantization


def train_network(training_data):

    net = Colorizer()

    # No gpu..
    net.cuda()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_X = train_X.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # criterion = nn.MSELoss()

    epochs = 1
    # batch_size = 100

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(training_data)):

            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            data = data.to(device)

            # print(data.shape)
            lab = split(data, [1, 2], dim=1)
            l = lab[0]
            ab = lab[1]

            # print(l.shape)
            # print(ab.shape)

            ab_onehot_list = [one_hot_quantization(v) for v in ab]

            ab_onehot = torch.Tensor(ab_onehot_list)

            l = Variable(l)
            ab_onehot = Variable(ab_onehot)

            out = net(l)

            optimizer.zero_grad()
            #loss = criterion(train, ab)
            loss = multi_class_cross_entropy_loss_torch(out, ab_onehot)
            loss.backward()
            optimizer.step()

            # if i == 10:
            #     print('loss is: ' + str(loss))
            #     break

            running_loss += (loss % 100)
        print(f'Running loss is: {running_loss}')

        torch.save(net.state_dict(), 'models/cifar10_colorizerCEL')


def multi_class_cross_entropy_loss_torch(predictions, labels):
    """
    Calculate multi-class cross entropy loss for every pixel in an image, for every image in a batch.

    In the implementation,
    - the first sum is over all classes,
    - the second sum is over all rows of the image,
    - the third sum is over all columns of the image
    - the last mean is over the batch of images.

    :param predictions: Output prediction of the neural network.
    :param labels: Correct labels.
    :return: Computed multi-class cross entropy loss.
    """

    print(predictions[0])
    print(predictions[0].shape)
    five_nearest_points, distances = find_k_nearest_q(predictions[h,w,q])


    loss = -torch.sum(torch.sum(torch.sum(labels *
                                          torch.log(predictions), dim=1), dim=1), dim=1)
    return loss


def load_data(data='cifar'):
    train_rgb, test_rgb = get_orig_data(data)

    training_labs, test_labs = get_lab_data(train_rgb, test_rgb)

    training_dataset = LabTrainingDataset(training_labs)
    # test_dataset = LabTestDataset(test_labs)

    lab_training_loader = DataLoader(training_dataset, batch_size=100,
                                     shuffle=True, num_workers=2)
    # lab_test_loader = DataLoader(test_dataset, batch_size=100,
    #                              shuffle=True, num_workers=2)

    torch.save(lab_training_loader, 'dataloaders/' +
               data+'_onehot_training.pth')
    # torch.save(lab_test_loader, 'dataloaders/'+data+'_onehot_test.pth')

    return lab_training_loader


if __name__ == '__main__':
    # load_data()
    train_data = torch.load('./dataloaders/cifar_lab_training_loader.pth')
    # test_data = torch.load('./dataloaders/cifar_lab_test_loader.pth')

    # train_data = torch.load('./dataloaders/cifar_onehot_training.pth')
    # test_data = torch.load('./dataloaders/cifar_lab_test_loader.pth')

    train_network(train_data)

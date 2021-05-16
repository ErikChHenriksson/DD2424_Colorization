from model import Colorizer
from preprocess import load_data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch import split
import torch
from tqdm import tqdm


def train_network(train_X, train_y):

    net = Colorizer()

    # No gpu..
    net.cuda()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_X = train_X.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 3
    # batch_size = 100

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_X)):

            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            data = data.to(device)

            # print(data.shape)
            lab = split(data, [1, 2], dim=1)
            l = lab[0]
            ab = lab[1]

            # print(l.shape)
            # print(ab.shape)

            l = Variable(l)
            ab = Variable(ab)

            train = net(l)

            optimizer.zero_grad()
            loss = criterion(train, ab)
            loss.backward()
            optimizer.step()

            # if i == 10:
            #     print('loss is: ' + str(loss))
            #     break

            running_loss += (loss % 100)
        print(f'Running loss is: {running_loss}')

        torch.save(net.state_dict(), 'models/cifar10_colorizer')


if __name__ == '__main__':
    train_X, train_y = load_data()
    train_network(train_X, train_y)

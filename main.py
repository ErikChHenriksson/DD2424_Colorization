from model import Colorizer
from preprocess import lab_training_loader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch import split


Colorizer = Colorizer()

# No gpu..
# Colorizer.cuda()

optimizer = optim.Adam(Colorizer.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 3
batch_size = 100

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(lab_training_loader):
        # print(data.shape)
        lab = split(data, [1, 2], dim=1)
        l = lab[0]
        ab = lab[1]

        # print(l.shape)
        # print(ab.shape)

        l = Variable(l)
        ab = Variable(ab)

        train = Colorizer(l)

        optimizer.zero_grad()
        loss = criterion(train, ab)
        loss.backward()
        optimizer.step()

        running_loss += (loss % 100)
    print(f'Running loss is: {running_loss}')


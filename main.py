from model import Colorizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np


def euclidian_loss(input, target):
    return torch.sqrt((input - target)**2).sum()/2


cnn = Colorizer()

optimizer = optim.Adam(cnn.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trainset = torch.load('./processed_data/trainset_lab_L_norm.pt')
labels = torch.load('./processed_data/ab_values5000.pt')
batch_size = 100
trainset = trainset[:5000]
num_batches = int(len(trainset) / batch_size)

epochs = 3

for epoch in range(epochs):
    running_loss = 0.0
    for b in range(num_batches):  # enumerate(trainset, 0):
        # get the inputs; data is a list of [inputs, labels]
        b_start = b*batch_size
        b_end = (b+1)*batch_size
        inputs, labels = trainset[b_start:b_end], labels[b_start:b_end]

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(np.array(labels))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(inputs)
        loss = euclidian_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print(f'loss: {running_loss}')
    running_loss = 0.0

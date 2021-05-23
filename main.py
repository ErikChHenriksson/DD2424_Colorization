from torch.utils.data.sampler import BatchSampler
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
from functions import find_k_nearest_q, one_hot_q, one_hot_quantization


def train_network(training_data):
    net = Colorizer()

    # No gpu..
    if torch.cuda.is_available():
        net.cuda()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    q_weights = torch.from_numpy(
        np.load('./quantized_space/w_points.npy')).to(device)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_X = train_X.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # criterion = nn.MSELoss()

    epochs = 7
    # batch_size = 100

    with open('loss.log', 'a') as f:
        for epoch in range(epochs):
            running_loss = 0.0
            f.write(f'Epoch {epoch}:\n')
            for i, data in enumerate(tqdm(training_data)):
                # data = data.to(device)

                # print(data.shape)
                # lab = split(data, [1, 2], dim=1)
                # l = lab[0]
                # ab = lab[1]

                # print(l.shape)
                # print(ab.shape)

                # ab_onehot_list = [one_hot_q(v).T for v in ab]

                # ab_onehot = torch.Tensor(ab_onehot_list).to(device)

                l = data[0]
                ab_onehot = data[1]

                # print(l.shape)
                # print(ab_onehot.shape)

                l = l.to(device)
                ab_onehot = ab_onehot.to(device)

                l = Variable(l)
                ab_onehot = Variable(ab_onehot)

                l = l.view(l.shape[0], -1, 32, 32)

                # print(l.view(l.shape[0], -1, 32, 32).shape)
                # print(ab_onehot.shape)

                out = net(l)

                optimizer.zero_grad()
                #loss = criterion(train, ab)
                loss = multi_class_cross_entropy_loss_torch(
                    out, ab_onehot, q_weights)
                loss.backward()
                optimizer.step()

                # if i == 5:
                #     # print('loss is: ' + str(loss))
                #     break
                f.write(f'Batch: {i}, loss: {round(float(loss), 4)}\n')

                running_loss += (loss % 100)
            print(f'Running loss is: {running_loss}')

            torch.save(net.state_dict(), 'models/cifar10_colorizerCEL')


def train_network_np(train_data, batch_size, epochs):
    net = Colorizer()

    # No gpu..
    if torch.cuda.is_available():
        net.cuda()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    q_weights = torch.from_numpy(
        np.load('./quantized_space/w_points.npy')).to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_X = torch.Tensor([i[0] for i in train_data]).view(-1, 32, 32)
    train_y = torch.Tensor([i[1] for i in train_data])

    with open('loss.log', 'a') as f:
        for e in range(epochs):
            for i in tqdm(range(0, len(train_X), batch_size)):
                batch_X = train_X[i:i +
                                  batch_size].view(-1, 1, 246, 32, 32).to(device)
                batch_y = train_y[i:i+batch_size].to(device)

                out = net(batch_X)

                optimizer.zero_grad()
                #loss = criterion(train, ab)
                loss = multi_class_cross_entropy_loss_torch(
                    out, batch_y, q_weights)
                loss.backward()
                optimizer.step()

                # if i == 5:
                #     # print('loss is: ' + str(loss))
                #     break
                f.write(f'Batch: {i}, loss: {round(float(loss), 4)}\n')

        torch.save(net.state_dict(), 'models/cifar10_colorizerCEL')

    return


def multi_class_cross_entropy_loss_torch(predictions, labels, q_weights):
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

    # print(predictions[0])
    # print(predictions[0].shape)
    # print(labels[0])
    # print(labels[0].shape)
    # five_nearest_points, distances = find_k_nearest_q(predictions[h,w,q])

    labels_t = labels.permute(0, 3, 2, 1)
    weighting_term = torch.max(q_weights * labels_t, dim=3)[0] * 1

    loss = -torch.mean(torch.sum(torch.sum(weighting_term * torch.sum(labels *
                                                                      torch.log(predictions), dim=1), dim=1), dim=1))

    # loss = nn.CrossEntropyLoss(predictions, labels)
    # print(loss)

    return loss


def load_data(data='cifar'):
    train_rgb, test_rgb = get_orig_data(data)

    training_labs = get_lab_data_train(train_rgb)
    # test_labs = get_lab_data_test(test_rgb)

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
    data = 'cifar'
    # train_data = torch.load('./dataloaders/cifar_lab_training_loader.pth')
    # # test_data = torch.load('./dataloaders/cifar_lab_test_loader.pth')

    # # train_data = torch.load('./dataloaders/cifar_onehot_training.pth')
    # # test_data = torch.load('./dataloaders/cifar_lab_test_loader.pth')

    train_data = torch.load('./dataloaders/' + data +
                            '_lab_training_loader_subset1000.pth')

    train_network(train_data)

    # train_data = np.load('./data_np/train_data5000.npy', allow_pickle=True)

    # train_network_np(train_data, 100, 3)

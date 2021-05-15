import torch
from model import Colorizer
from preprocess import lab_test_loader
from torch import split
from torch.autograd import Variable
from skimage import color
from PIL import Image

# get trained model
model = Colorizer()
model.load_state_dict((torch.load('models/cifar10_colorizer')))
model.eval()

for i, data in enumerate(lab_test_loader):

    # get the first picture in batch
    orig = data[0, :, :, :].data.cpu().numpy().T

    # save orig image
    im = Image.fromarray(color.lab2rgb(orig), mode='RGB')
    im.save('orig.png', 'PNG')

    # get l dimension of orig img
    lab = split(data, [1, 2], dim=1)
    l = lab[0]
    l = split(l, [1, 99], dim=0)
    l = l[0] # get the first image
    print(l.shape)
    l = Variable(l)

    # predict ab
    ab = model(l)
    out = torch.cat((l, ab), dim=1)

    # transform to rgb and save the img
    rgb = color.lab2rgb(out.data.cpu().numpy()[0,...].T)
    print(rgb.shape)
    im = Image.fromarray(rgb, mode='RGB')
    im.save('rgb.png', 'PNG')

    break

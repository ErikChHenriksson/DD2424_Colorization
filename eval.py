import torch
from model import Colorizer
from preprocess import load_data
from torch import split
from torch.autograd import Variable
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # get trained model
    model = Colorizer()
    # You can remove the map_location argument if you want to use GPU
    model.load_state_dict(
        (torch.load('models/cifar10_colorizerCEL', map_location=torch.device('cpu'))))
    model.eval()

    train_X, train_y = load_data()
    case, num_cases = 1, 10
    for i, data in enumerate(train_X):
        # get the first picture in batch
        orig = data[0, :, :, :].data.cpu().numpy().T

        # save orig image
        im = Image.fromarray(color.lab2rgb(orig), mode='RGB')
        im.save('orig.png', 'PNG')

        # get l dimension of orig img
        lab = split(data, [1, 2], dim=1)
        l = lab[0]
        l = split(l, [1, 99], dim=0)
        l = l[0]  # get the first image
        l = Variable(l)

        # predict ab
        ab = model(l)
        print('colormax', torch.max(ab))
        out = torch.cat((l, ab), dim=1)

        # transform to rgb and save the img
        rgb = color.lab2rgb(out.data.cpu().numpy()[0, ...].T)
        #print(rgb.shape)
        im = Image.fromarray(rgb, mode='RGB')
        im.save('rgb.png', 'PNG')

        # plot images
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title('Original')
        ax2.set_title('Network output')
        ax1.imshow(color.lab2rgb(orig))
        ax2.imshow(rgb)
        fig.savefig('imgs/test'+str(case)+'.png')
        case = case + 1
        print('case=', case, 'num_cases=', num_cases)
        if case > num_cases:
            break

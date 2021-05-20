import torch
from model_cifar import Colorizer
from preprocess import *
from torch import split
from torch.autograd import Variable
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
from functions import *


def load_data(data='cifar'):
    train_rgb, test_rgb = get_orig_data(data)

    training_labs = get_lab_data_train(train_rgb)
    test_labs = get_lab_data_train(test_rgb)

    training_dataset = LabTrainingDataset(training_labs)
    test_dataset = LabTestDataset(test_labs)

    lab_training_loader = DataLoader(training_dataset, batch_size=100,
                                     shuffle=True, num_workers=2)
    lab_test_loader = DataLoader(test_dataset, batch_size=100,
                                 shuffle=True, num_workers=2)

    torch.save(lab_training_loader, 'dataloaders_eval/' +
               data+'_lab_training_loader.pth')
    torch.save(lab_training_loader, 'dataloaders_eval/' +
               data+'_lab_test_loader.pth')

    return lab_training_loader, lab_test_loader


def save_output_imgs(model, test_data):
    q_points = np.load('./quantized_space/q_points.npy')

    case, num_cases = 1, 10
    for i, data in enumerate(test_data):
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
        q_dist = model(l)

        ab = q_distribution_to_ab(q_dist[0], q_points)
        ab_t = torch.Tensor(ab).T.view(1, 2, 32, 32)

        print('colormax', torch.max(ab_t))
        out = torch.cat((l, ab_t), dim=1)

        # transform to rgb and save the img
        rgb = color.lab2rgb(out.data.cpu().numpy()[0, ...].T)
        # print(rgb.shape)
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


if __name__ == '__main__':
    # get trained model
    model = Colorizer()
    # You can remove the map_location argument if you want to use GPU
    model.load_state_dict(
        (torch.load('models/cifar10_colorizerCEL', map_location=torch.device('cpu'))))
    model.eval()

    # train_data = torch.load('./dataloaders_eval/cifar_lab_training_loader.pth')
    test_data = torch.load('./dataloaders_eval/cifar_lab_test_loader.pth')

    save_output_imgs(model, test_data)

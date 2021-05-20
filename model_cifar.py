import torch.nn as nn
from torch.nn import BatchNorm2d, ReLU, Sequential, Conv2d, Softmax, Upsample


class Colorizer(nn.Module):
    def __init__(self):
        super().__init__()

        self._l_cent = 50.
        self._l_norm = 100.
        self._ab_norm = 110.

        model1 = [
            # CONV1
            Conv2d(1, 8, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(8, 8, kernel_size=3, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(8)]
        # CONV2
        model2 = [
            Conv2d(8, 16, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(16, 16, kernel_size=3, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(16)]
        # CONV3
        model3 = [
            Conv2d(16, 32, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(32)]
        # CONV4
        model4 = [
            Conv2d(32, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64)]
        # CONV5
        model5 = [
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64)]
        # CONV6
        model6 = [
            Conv2d(64, 64, kernel_size=3, padding=2),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64)]
        # CONV7
        model7 = [
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64)]
        # CONV8
        model8 = [
            Conv2d(64, 32, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(inplace=True),
            # TO COLOR PROBABILITIES
            Conv2d(32, 208, kernel_size=3, padding=0)]

        self._model1 = Sequential(*model1)
        self._model2 = Sequential(*model2)
        self._model3 = Sequential(*model3)
        self._model4 = Sequential(*model4)
        self._model5 = Sequential(*model5)
        self._model6 = Sequential(*model6)
        self._model7 = Sequential(*model7)
        self._model8 = Sequential(*model8)

        self._softmax = Softmax(dim=1)
        self._final = Conv2d(208, 2, kernel_size=1,
                             stride=1, padding=0, dilation=1, bias=False)
        self._up = Upsample(scale_factor=1, mode='bilinear')

    def forward(self, input):
        # print(input.shape)
        conv1 = self._model1(self.norm_l(input))
        conv2 = self._model2(conv1)
        conv3 = self._model3(conv2)
        conv4 = self._model4(conv3)
        conv5 = self._model5(conv4)
        conv6 = self._model6(conv5)
        conv7 = self._model7(conv6)
        conv8 = self._model8(conv7)

        # out = self._final(self._softmax(conv8))
        # return self.unnorm_ab(self._up(out))
        return self._softmax(conv8)

    def norm_l(self, l):
        return (l - self._l_cent) / self._l_norm

    def unnorm_l(self, l):
        return l * self._l_norm + self._l_cent

    def norm_ab(self, ab):
        return ab / self._ab_norm

    def unnorm_ab(self, ab):
        return ab * self._ab_norm

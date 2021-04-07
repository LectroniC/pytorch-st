"""Model Implementation for "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

torch.manual_seed(0)
### image transformation network ###


class ResidualBlock(nn.Module):
    """residual block used in style transfer net"""

    def __init__(self, channels=3):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv_1 = nn.Conv2d(self.channels, self.channels,
                                kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.in_1 = nn.InstanceNorm2d(self.channels)
        self.conv_2 = nn.Conv2d(self.channels, self.channels,
                                kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.in_2 = nn.InstanceNorm2d(self.channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """input: [N, C, H, W]"""
        residual = x
        x = self.relu(self.in_1(self.conv_1(x)))
        x = self.in_2(self.conv_2(x))
        x = x + residual
        return x


class UpsampleConvLayer(torch.nn.Module):
    """some research shows it works better than transpose conv: """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=self.upsample)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=kernel_size//2, padding_mode='reflect')

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.conv2d(x)
        return x


class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        self.ref_pad = nn.ReflectionPad2d(40)
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=9,
                                stride=1, padding=4, padding_mode='reflect')
        self.in_1 = nn.InstanceNorm2d(32)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3,
                                stride=2, padding=1, padding_mode='reflect')
        self.in_2 = nn.InstanceNorm2d(64)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3,
                                stride=2, padding=1, padding_mode='reflect')
        self.in_3 = nn.InstanceNorm2d(128)

        self.res_1 = ResidualBlock(128)
        self.res_2 = ResidualBlock(128)
        self.res_3 = ResidualBlock(128)
        self.res_4 = ResidualBlock(128)
        self.res_5 = ResidualBlock(128)

        # self.conv_t_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_1 = UpsampleConvLayer(
            128, 64, kernel_size=3, stride=1, upsample=2)
        self.in_4 = nn.InstanceNorm2d(64)
        # self.conv_t_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_2 = UpsampleConvLayer(
            64, 32, kernel_size=3, stride=1, upsample=2)
        self.in_5 = nn.InstanceNorm2d(32)
        self.conv_f = nn.Conv2d(32, 3, kernel_size=9,
                                stride=1, padding=4, padding_mode='reflect')

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """ input:  [N, 3, 256, 256]
            output: [N, 3, 256, 256]
        """
        x = self.relu(self.in_1(self.conv_1(x)))
        x = self.relu(self.in_2(self.conv_2(x)))
        x = self.relu(self.in_3(self.conv_3(x)))
        x = self.res_5(self.res_4(self.res_3(self.res_2(self.res_1(x)))))
        x = self.relu(self.in_4(self.up_1(x)))
        x = self.relu(self.in_5(self.up_2(x)))
        x = self.tanh(self.conv_f(x))
        return x

### vgg16 network for loss computation ###


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        feats = models.vgg16(pretrained=True).features
        self.slice1, self.slice2, self.slice3, self.slice4 = nn.Sequential(
        ), nn.Sequential(), nn.Sequential(), nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), feats[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), feats[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), feats[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), feats[x])

        for para in self.parameters():
            para.requires_grad = False

    def forward(self, x):
        """ input:  [N, 3, 256, 256]
            output: [N, 64, 256, 256], [N, 128, 128, 128], [N, 256, 64, 64], [N, 512, 32, 32]
        """
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x
        return (relu1_2, relu2_2, relu3_3, relu4_3)

# [0] 12 [1] 22 [2] 33 [3] 43
# feature reconstruction loss at layer relu2_2
# style reconstruction loss at layer relu1_2, relu2_2, relu3_3, and relu4_3


def L_feat(feats, targets):
    """ compute feature reconstruction loss
        input: [C, H, W]
        output: scalar tensor
    """
    criterion = nn.MSELoss(reduction='mean')
    return criterion(feats, targets)


def gram_matrix(input):
    """find gram matrix: this returns a batch matrix"""
    b, c, h, w = input.size()
    features = input.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(h * w)


def L_style(feats, targets):
    """ compute style reconstruction loss (for single input/target pair)
        input: [C, H, W]
        output: scalar tensor
    """
    criterion = nn.MSELoss(reduction='mean')
    return criterion(gram_matrix(feats), gram_matrix(targets))


def L_pixel(feats, targets):
    """ Pixel-wise loss: used when have groundtruth; typically not used
    """
    criterion = nn.MSELoss(reduction='mean')
    return criterion(feats, targets)


def L_tv(preds):
    """ total variance regularization loss
    """
    loss = torch.mean(torch.abs(preds[:, :, :, :-1] - preds[:, :, :, 1:])) + \
        torch.mean(torch.abs(preds[:, :, :-1, :] - preds[:, :, 1:, :]))
    return loss


def test_net():
    x = torch.randn(20, 3, 256, 256)
    print(x.shape)
    model = StyleTransferNet()
    out = model(x)
    print(out.shape)


def test_vgg():
    x = torch.randn(20, 3, 256, 256)
    mdl = Vgg16()
    print(mdl)
    out = mdl(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)


def test_loss():
    x = torch.randn(20, 3, 256, 256)
    model = StyleTransferNet()
    loss_net = Vgg16()
    out = model(x)
    f1, f2, f3, f4 = loss_net(x)
    p1, p2, p3, p4 = loss_net(out)
    l1 = L_feat(p2, f2)
    print(l1)
    l2 = L_style(p1, f1)
    print(l2)
    l3 = L_tv(out)
    print(l3)


# A class wrapper is better to avoid recalculation of style_features
class Loss_plst():
    def __init__(vgg, style_img, lambda_c=1e0, lambda_s=1e5,  lambda_tv=1e-7):
        # Style image
        pass

    def extract_and_calculate_loss(self):
        # TODO: Wrap the loss function.
        # Return content_loss, style_loss, tv_loss
        pass


def main():
    test_net()
    test_vgg()
    test_loss()


if __name__ == "__main__":
    main()

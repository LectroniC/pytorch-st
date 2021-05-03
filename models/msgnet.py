""" Model Implementation for "Multi-style Generative Network for Real-time Transfer" """

import numpy as np
import torch 
import torch.nn as nn
import torchvision.models as models

torch.manual_seed(0)

##################################### main MSG model #####################################

def gram_matrix(input):
    """find gram matrix: this returns a batch matrix: we also divide by C here as well to match paper"""
    b, c, h, w = input.size()
    features = input.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(c * h * w)

class CoMatchLayer(nn.Module):
    def __init__(self, num_channel, batch_size=1):
        super(CoMatchLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, num_channel, num_channel))
        self.C = num_channel
        self.G = torch.Tensor(batch_size, self.C, self.C)
        self.reset_parameters()

    def reset_parameters(self):
        """reset parameter following original paper"""
        self.weight.data.uniform_(0.0, 0.02)

    def set_target(self, target):
        """ target: G(Fi(x_s)) from Gram calculation: [B, C, C]"""
        self.G = target

    def forward(self, x):
        """X: Fi(x_c): [B, C, H, W]"""
        # step1: calculate WG: [B, C, C]
        P = torch.bmm(self.weight.expand_as(self.G), self.G)
        # step2: calculate [phi^TWG]^T = P^Tphi: [B, C, C] [B, C, HW] = [B, C, HW]
        R = torch.bmm(P.transpose(1, 2).expand(x.size(0), self.C, self.C), x.view(x.size(0), x.size(1), -1))
        # step3: return phi^(-1)R: separate out H, W in last dim
        return R.view_as(x)

class UpsampleConvLayer(torch.nn.Module):
    """some research shows it works better than transpose conv: same upsampleConv used in PLST"""

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

class DownSampleResidualBlock(nn.Module):
    """downsample residual block used in msg net"""
    def __init__(self, in_channels, out_block_channels, stride=1, downsample=None):
        super(DownSampleResidualBlock, self).__init__()
        self.downsample = downsample
        if self.downsample is not None:
            self.res_layer = nn.Conv2d(in_channels, out_block_channels*4, kernel_size=1, stride=stride)
        self.conv_1 = nn.Conv2d(in_channels, out_block_channels, kernel_size=1, stride=1)
        self.in_1 = nn.InstanceNorm2d(out_block_channels)
        self.conv_2 = nn.Conv2d(out_block_channels, out_block_channels, kernel_size=3, stride=stride, padding=1, padding_mode='reflect')
        self.in_2 = nn.InstanceNorm2d(out_block_channels)
        self.conv_3 = nn.Conv2d(out_block_channels, out_block_channels*4, kernel_size=1, stride=1)
        self.in_3 = nn.InstanceNorm2d(out_block_channels*4)
        self.relu = nn.ReLU()

    def forward(self, x):
        """input: [N, C, H, W]"""
        if self.downsample is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        x = self.relu(self.in_1(self.conv_1(x)))
        x = self.relu(self.in_2(self.conv_2(x)))
        x = self.in_3(self.conv_3(x))
        x = x + residual
        return x

class UpSampleResidualBlock(nn.Module):
    """upsample residual block used in msg net"""
    def __init__(self, in_channels, out_block_channels, stride=2):
        super(UpSampleResidualBlock, self).__init__()
        self.res_layer = UpsampleConvLayer(in_channels, out_block_channels*4, kernel_size=1, stride=1, upsample=stride)
        self.conv_1 = nn.Conv2d(in_channels, out_block_channels, kernel_size=1, stride=1)
        self.in_1 = nn.InstanceNorm2d(out_block_channels)
        self.conv_2 = UpsampleConvLayer(out_block_channels, out_block_channels, kernel_size=3, stride=1, upsample=stride)
        self.in_2 = nn.InstanceNorm2d(out_block_channels)
        self.conv_3 = nn.Conv2d(out_block_channels, out_block_channels*4, kernel_size=1, stride=1)
        self.in_3 = nn.InstanceNorm2d(out_block_channels*4)
        self.relu = nn.ReLU()

    def forward(self, x):
        """input: [N, C, H, W]"""
        residual = self.res_layer(x)
        x = self.relu(self.in_1(self.conv_1(x)))
        x = self.relu(self.in_2(self.conv_2(x)))
        x = self.in_3(self.conv_3(x))
        x = x + residual
        return x

class MSGNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, block_size=64, n_blocks=6):
        super(MSGNet, self).__init__()

        self.down_model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            DownSampleResidualBlock(64, 32, 2, 1),
            DownSampleResidualBlock(32*4, block_size, 2, 1)
        )
    
        model = []
        self.co_match = CoMatchLayer(block_size*4)
        model += [self.down_model]
        model += [self.co_match]

        for i in range(n_blocks):
            model += [DownSampleResidualBlock(block_size*4, block_size, 1, None)]

        self.up_model = nn.Sequential(
            UpSampleResidualBlock(block_size*4, 32, 2),
            UpSampleResidualBlock(32*4, 16, 2),
            nn.InstanceNorm2d(16*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16*4, output_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        )
        model += [self.up_model]

        self.model = nn.Sequential(*model)

    def set_target(self, Xs):
        f = self.down_model(Xs)
        G = gram_matrix(f)
        self.co_match.set_target(G)

    def forward(self, x):
        return self.model(x)

##################################### vgg16 network for loss computation #####################################

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


def L_feat(feats, targets):
    """ compute feature reconstruction loss
        input: [B, C, H, W]
        output: scalar tensor
    """
    criterion = nn.MSELoss(reduction='mean')
    return criterion(feats, targets)


def gram_matrix_loss(input):
    """find gram matrix: this returns a batch matrix"""
    b, c, h, w = input.size()
    features = input.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(c * h * w)


def L_style(feats, targets):
    """ compute style reconstruction loss (for single input/target pair)
        input: [B, C, H, W]
        output: scalar tensor
    """
    criterion = nn.MSELoss(reduction='mean')
    return criterion(gram_matrix_loss(feats), gram_matrix_loss(targets))


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
    x_content = torch.randn(20, 3, 256, 256)
    x_style = torch.randn(1, 3, 256, 256)
    print(x_content.shape, x_style.shape)
    model = MSGNet()
    model.set_target(x_style)
    out = model(x_content)
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
    x_content = torch.randn(20, 3, 256, 256)
    x_style = torch.randn(1, 3, 256, 256)
    model = MSGNet()
    loss_net = Vgg16()
    model.set_target(x_style)
    out = model(x_content)
    f1, f2, f3, f4 = loss_net(x)
    p1, p2, p3, p4 = loss_net(out)
    l1 = L_feat(p2, f2)
    print(l1)
    l2 = L_style(p1, f1)
    print(l2)
    l3 = L_tv(out)
    print(l3)

# A class wrapper is better to avoid recalculation of style_features
class Loss_msg():
    def __init__(self, vgg, lambda_c=1.0, lambda_s=1.0,  lambda_tv=1.0):
        # Style image
        self.vgg = vgg
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.lambda_tv = lambda_tv

    def update_style_feats(self, style_img, batch_size):
        # need to be called before loss calculation
        self.style_img = style_img
        self.style_relu1_2, self.style_relu2_2, self.style_relu3_3, self.style_relu4_3 = self.vgg(self.style_img)
        self.style_relu1_2 = self.style_relu1_2.repeat(batch_size, 1, 1, 1)
        self.style_relu2_2 = self.style_relu2_2.repeat(batch_size, 1, 1, 1)
        self.style_relu3_3 = self.style_relu3_3.repeat(batch_size, 1, 1, 1)
        self.style_relu4_3 = self.style_relu4_3.repeat(batch_size, 1, 1, 1)
        
    def extract_and_calculate_loss(self, x, y_hat):
        # TODO: Wrap the loss function.
        # Return content_loss, style_loss, tv_loss
        _, _, content_target, _ = self.vgg(x)
        content_relu1_2, content_relu2_2, content_relu3_3, content_relu4_3 = self.vgg(y_hat)
        loss_c = L_feat(content_relu3_3, content_target)
        loss_s = L_style(content_relu1_2, self.style_relu1_2) + L_style(content_relu2_2, self.style_relu2_2)+ \
            L_style(content_relu3_3, self.style_relu3_3) + L_style(content_relu4_3, self.style_relu4_3)
        loss_tv = L_tv(y_hat)
        return self.lambda_c * loss_c, self.lambda_s * loss_s, self.lambda_tv * loss_tv


if __name__ == "__main__":
    test_net()
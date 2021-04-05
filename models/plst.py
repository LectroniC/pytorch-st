"""Model Implementation for "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"""

import numpy as np
import torch 
import torch.nn as nn


## image transformation network ##

class ResidualBlock(nn.Module):
    """residual block used in style transfer net"""
    def __init__(self, channels=3):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv_1 = nn.Conv2d(self.channels, self.channels, kernel_size=3)
        self.in_1 = nn.InstanceNorm2d(self.channels)
        self.conv_2 = nn.Conv2d(self.channels, self.channels, kernel_size=3)
        self.in_2 = nn.InstanceNorm2d(self.channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """input: [N, C, H, W]"""
        residual = x[:,:,2:-2,2:-2]
        out = self.relu(self.in_1(self.conv_1(x)))
        out = self.in_2(self.conv_2(out))
        out = out + residual
        return out


class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        self.ref_pad = nn.ReflectionPad2d(40)
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4, padding_mode='reflect')
        self.in_1 = nn.InstanceNorm2d(32)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.in_2 = nn.InstanceNorm2d(64)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.in_3 = nn.InstanceNorm2d(128)

        self.res_1 = ResidualBlock(128)
        self.res_2 = ResidualBlock(128)
        self.res_3 = ResidualBlock(128)
        self.res_4 = ResidualBlock(128)
        self.res_5 = ResidualBlock(128)

        self.conv_t_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in_4 = nn.InstanceNorm2d(64)
        self.conv_t_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in_5 = nn.InstanceNorm2d(32)
        self.conv_f = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4, padding_mode='reflect')

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        """input: [N, 3, 256, 256]"""
        x = self.ref_pad(x)
        x = self.relu(self.in_1(self.conv_1(x)))
        x = self.relu(self.in_2(self.conv_2(x)))
        x = self.relu(self.in_3(self.conv_3(x)))
        x = self.res_5(self.res_4(self.res_3(self.res_2(self.res_1(x)))))
        x = self.relu(self.in_4(self.conv_t_1(x)))
        x = self.relu(self.in_5(self.conv_t_2(x)))
        x = self.tanh(self.conv_f(x))
        return x


def test():
    x = torch.randn(20, 3, 256, 256)
    print(x.shape)
    model = StyleTransferNet()
    out = model(x)
    print(out.shape)


def main():
    test()


if __name__ == "__main__":
    main()


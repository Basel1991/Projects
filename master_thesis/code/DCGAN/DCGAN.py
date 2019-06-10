"""
code derived from Pytorch tutorials <https://github.com/inkawhich>`
modified and improved by Basel Alyafi
year: 2019
"""

from __future__ import print_function

import torch.nn as nn

# custom weights initialization called on netG and netD

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

######################################################################
# Generator
# ~~~~~~~~~

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):

        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 10, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 10),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 10, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. (ngf) x 64 X 64

            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 X 128
        )

    def forward(self, input):
        return self.model(input)

    # a path that can be used for saving the model
    Gpath = '/home/basel/PycharmProjects/DCGAN/models/Generators/'

######################################################################
# Discriminator

class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc ):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            # input size is 128 X 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (nc) x 64 x 64

            nn.Conv2d(ndf, ndf * 2,      kernel_size=6, stride=2, padding=2, bias=False), #was 4,2,1
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32

            nn.Conv2d(ndf * 2, ndf * 4,  kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 4, ndf * 8,  kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 8, ndf * 10, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ndf * 10),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 10, 1,       kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    Dpath = '/home/basel/PycharmProjects/DCGAN/models/Discriminators/'

######################################################################

class PSGenerator(nn.Module):
    def __init__(self, L, nz, ngf, nc):
        super(PSGenerator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.L = L

        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*16),
            nn.LeakyReLU(),

            nn.PixelShuffle(2),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(),

            nn.PixelShuffle(2),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),

            nn.PixelShuffle(2),
            nn.BatchNorm2d(int(ngf/4)),
            nn.LeakyReLU(),

            nn.PixelShuffle(2),
            nn.BatchNorm2d(int(ngf / 16)),
            nn.LeakyReLU(),

            nn.PixelShuffle(2),
            nn.Tanh())

    def forward(self, *input):
        return self.model(*input)

import glob
from skimage import io
from torch import nn
import torch
import os
import numpy as np

class Classifier(nn.Module):

    def __init__(self, **keyargs):
        super(Classifier, self).__init__()

        channels = keyargs['channels']
        kernels = keyargs['kernels']

        paddings = keyargs['paddings']
        dropout = keyargs['dropout']
        self.path = os.path.join(os.getcwd(), 'Evaluators', keyargs['id'])

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=kernels[0], stride=2, padding=paddings[0]),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(negative_slope=2e-1, inplace=True),

            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernels[1], stride=2, padding=paddings[1]),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(negative_slope=2e-1, inplace=True),

            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernels[2], stride=2, padding=paddings[2]),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(negative_slope=2e-1, inplace=True),

            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=kernels[3], stride=2, padding=paddings[3]),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU(negative_slope=2e-1, inplace=True),


            nn.Conv2d(in_channels=channels[3], out_channels=channels[4], kernel_size=kernels[4], stride=2, padding=paddings[4]),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(channels[4]),
            nn.LeakyReLU(negative_slope=2e-1, inplace=True),

            nn.Conv2d(in_channels=channels[4], out_channels=channels[5], kernel_size=kernels[5], stride=1, padding=paddings[5]),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(channels[5]),
            nn.LeakyReLU(negative_slope=2e-1, inplace=True),
        )

    def forward(self, input):
        return self.model(input)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc, id):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.id = str(id)
        self.path = os.path.join(os.getcwd(), 'Evaluation', 'Evaluators', str(id))
        self.main = nn.Sequential(

            # input size is 128 X 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (nc) x 64 x 64

            nn.Conv2d(ndf, ndf * 2, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 8, ndf * 10, kernel_size=6, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ndf * 10),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 10, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, input):
        return self.main(input)

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_names, indices, transforms, label, shuffle_seed=-1):

        if shuffle_seed != -1:
            np.random.seed(shuffle_seed)
            np.random.shuffle(imgs_names)

        self.names = (imgs_names[indices])
        self.transforms = transforms
        self.label = label

    def __getitem__(self, index):

        img = io.imread(self.names[index])
        if self.transforms is not None:
            img = self.transforms(img)

        return (img, self.label)

    def __len__(self):
        return len(self.names)




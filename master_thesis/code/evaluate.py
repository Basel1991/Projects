"""
author: Basel Alyafi
year: 2019
Erasmus Mundus in Medical Imaging and Applications (MAIA) 2017-2019
Master Thesis

This script is to evaluate the trained DCGAN by augmenting the minority class in an unbalanced classification problem.
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import make_grid
from torchvision import models, transforms

from Evaluation.classes import Classifier, Discriminator, ImageDataset
from Evaluation.helpers import data_loaders, train_model, initialize_weights, predict, set_parameter_requires_grad, data_loader
import numpy as np

import sys
from helpers import args_parser
from termcolor import cprint

#------------------------------------------------------------------------ init
parser = args_parser(epochs=int, lr=float, fold=int, data_size=int, mode=str, aug=int)
args = parser.parse_args()
fold_idx = args.fold
data_size = args.data_size
mode = args.mode
# AF=args.AF

# Path to parent folders of training/val/test subfolders, each subfolder is for a class
train_path = '/home/user/training_images/...'

val_path = '/home/user/val_images/...'
test_path = '/home/user/test_images/...'

# True, False
save_model  = True
load_saved  = False
train       = True

samples = 64
device = Classifier.device
shuffle=True
plot = True

# Hyper Parameters
epochs = args.epochs
print_every = 2
validate_every = 10
ndf = 9

learning_rate = args.lr

model_id = 'DFCN_1505_3fold' + str(fold_idx) + '_9911_weigDec_dataSize_{}_'.format(data_size) + mode +'_CAug{}_OptimalSynthSize_{}'.format(args.aug, AF) #TODO delete the last part
# model_id = 'DFCN_1305_fold' + str(fold_idx) + '_692115_weigDec'
configuration = 'batsiz_{}---lr_{}---fm_{}---epochs_{}---id_{}---dataSize_{}.png'.format(samples, learning_rate, ndf, epochs, model_id, data_size)
print('Configuration:\n' + configuration)
#------------------------------------------------------------------------------------------------------------------------
# Network parameters
loss = nn.BCEWithLogitsLoss()

test_transform = transforms.Compose([
                                            transforms.Grayscale(),
                                            transforms.Resize([128, 128]),
                                            transforms.ToTensor()
])

if args.aug:
    train_transform = transforms.Compose([
                                            transforms.Grayscale(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            # transforms.ColorJitter(.05, .05 , .05, .05),
                                            transforms.Resize([128, 128]),
                                            transforms.ToTensor()
                                        ])
else:
    train_transform = test_transform

train_loader = data_loader(path=train_path, batch_size=samples, shuffle=shuffle, transforms=train_transform)
val_loader = data_loader(path=val_path, batch_size=samples, shuffle=False, transforms=test_transform)
test_loader = data_loader(path=test_path, batch_size=samples, shuffle=False, transforms=test_transform)

model = Discriminator(ngpu=1, ndf=ndf, nc=1, id=model_id)

model.to(device)

# Optimizer selection
real_optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=.0005)
fake_optimizer = optim.Adam(params=model.parameters(), lr= learning_rate, betas=(0.9, 0.99), weight_decay=.0005)

real_scheduler = optim.lr_scheduler.StepLR(optimizer=real_optimizer, step_size=1, gamma=0.95)
fake_scheduler = optim.lr_scheduler.StepLR(optimizer=fake_optimizer, step_size=1, gamma=0.95)

# either load a saved model or initialize with Xavier
if load_saved:
    if os.path.exists(model.path):
        model.load_state_dict(torch.load(os.path.join(model.path, model.id)))
else:
    model.apply(initialize_weights)

print('size of train {}, val {}, test loaders {}'.format(len(train_loader), len(val_loader), len(test_loader)))

# compute the number of parameters
trainable_parameters = np.sum(param.numel() for param in model.parameters() if param.requires_grad)
num_param = np.sum(param.numel() for param in model.parameters())

print('Summary \n', model, '\n  parameters: {}, trainable {}'.format(trainable_parameters, num_param) )
imgs, labels = next(iter(test_loader))

# train, val and test
f1=0
if train:

    model, f1 = train_model(model=model, data_loader={'train': train_loader, 'val': val_loader}, optimizer=real_optimizer, scheduler=real_scheduler, epoch_num=epochs,
                            print_every=print_every,
                            validate_every=validate_every, pos_weight =1, save=save_model, initial_f1=f1, plot=plot)

cprint('\n\nval Results:', 'yellow', attrs=['bold'])
predict(model, val_loader, pos_loss_weight=5, verbose=3, tag='_val')

cprint('\n\nTest Results:', 'red', attrs=['bold'])
predict(model, test_loader, pos_loss_weight=5, verbose=3, tag='_test', xlfile_name='/home/basel/Documents/Thesis/OptimalSynthSize.xlsx', prefix=[data_size, AF, fold_idx ])

print('\ndata loaders length (train {}, val {}, test {})'.format(len(train_loader.dataset.imgs), \
                                                               len(val_loader.dataset.imgs), len(test_loader.dataset.imgs)))
print('\na batch\'s dimensions' , imgs.size())
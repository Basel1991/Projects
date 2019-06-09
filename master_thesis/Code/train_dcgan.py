"""
Code adapted from Pytorch tutorials <https://github.com/inkawhich>`
Modified by Basel Alyafi
year: 2019

This script is used to train the DCGAN.
"""
from __future__ import print_function
# %matplotlib inline
import os
import time

import matplotlib
import seaborn
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from Evaluation.pytorch_fid.fid_score import calculate_fid_given_paths
from Gans.DCGAN import Generator, Discriminator, weights_init
import datetime
from termcolor import colored, cprint

from helpers import run_generator, args_parser,plot_fig
matplotlib.use('Agg') #TODO this backend does NOT show figures

if __name__ == '__main__':

    # parse the arguments to ease inputting them through the console
    parser = args_parser(epochs=int, ndf=int, ngf=int, lr_D=float, lr_G=float, fold=int, data_size=int)
    args = parser.parse_args()

    #for pytorch synchronization issues with GPU
    torch.backends.cudnn.benchmark=True

    dataset_size = args.data_size
    # define all needed paths

    # the path to the parent folder of all classes. Each subfolder contains the training images for one class
    dataroot =  '~/home/user/dataset'

    # the path where a sample of the training images that will be used to compute FID
    resized_RGB_dataroot = '/home/user/real_sample'

    today = datetime.datetime.today()
    minute, hour, day, month, year = today.minute, today.hour, today.day, today.month, today.year

    G_model_name = 'baseG_{:02d}.{:02d}_{:02d}_{:02d}_{:02d}_FID'.format(hour, minute,day, month, year)
    D_model_name = 'baseD_{:02d}.{:02d}_{:02d}_{:02d}_{:02d}_FID'.format(hour, minute, day, month, year)

    # the path where to save figures + fake images + models
    save_fig_path = '/home/user/DCGAN/' + G_model_name + '/'
    save_fake = save_fig_path + 'fake_images/'
    note = 'nz200_MassMalLes_fm10_fold{}_size{}_.99Step_oneSidedSmooth_Dk6p2_2345_resizeBoth'.format(args.fold, dataset_size) #TODO was allLes

    # to save the model
    save_model = True

    # Number of workers for dataloader
    workers = 8
    batch_size = 64

    ######################################################################
    #hyper parameters

    # Spatial size of training images. All images will be resized to this
    image_size = 128
    # Number of channels
    nc = 1
    # Size of z latent vector ( size of generator input)
    nz = 200
    # feature maps size initialization, ndf for discriminator, ngf for generator
    ndf, ngf = args.ndf, args.ngf

    num_epochs = args.epochs
    lr_D = args.lr_D
    lr_G = args.lr_G

    beta1 = 0  # for Adam optimizer
    ngpu = 1    # number of GPUs

    fid_every = 10  # calc FID every
    print_every = 2 #print losses every

    configuration = 'BS_{}---lr_{}_{}---fm_{}_{}---e_{}---B1_{}---{}.svg'.format(batch_size, lr_D, lr_G, ndf, ngf, num_epochs, beta1, note)
    print('Configuration:\n' + configuration)
    ######################################################################
    # Dataset

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize([image_size, image_size]),
                                   # transforms.CenterCrop(image_size),

                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),

                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5])
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print('device', device)

    ######################################################################
    # Generator

    netG = Generator(ngpu, nz, ngf, nc).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the network summary
    trainable_parameters = np.sum(param.numel() for param in netG.parameters() if param.requires_grad)
    num_param = np.sum(param.numel() for param in netG.parameters())

    print('Generator Summary \n', netG, '\n  parameters: {}, trainable {} \n model ID {}'.format(trainable_parameters, num_param, G_model_name ))
    ######################################################################
    # Discriminator

    netD = Discriminator(ngpu, ndf, nc).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the network summary
    trainable_parameters = np.sum(param.numel() for param in netD.parameters() if param.requires_grad)
    num_param = np.sum(param.numel() for param in netD.parameters())

    print('Discriminator Summary \n', netD, '\n  parameters: {}, trainable {} \n model id {}'.format(trainable_parameters, num_param, D_model_name) )

    ######################################################################
    # Initialization

    criterion = nn.BCELoss()

    # Create batch of latent vectors that will be used to visualize the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))

    # learning rate schedulers
    scheduler_D = optim.lr_scheduler.StepLR(optimizer=optimizerD, step_size=8, gamma=0.99)
    scheduler_G = optim.lr_scheduler.StepLR(optimizer=optimizerG, step_size=10, gamma=0.99)

    ######################################################################
    # Training

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    FIDs=[]
    iters = 0

    # FID initialization
    min_fid=5000

    print("Starting Training Loop...")
    start = time.time()

    # For each epoch
    for epoch in range(num_epochs):

        ##schedulers passed by here
        scheduler_D.step()
        scheduler_G.step()

        # take a random value every epoch,  0.7< real < 1.0: One-sided smoothing
        real_label = 0.3 * np.random.random_sample() + 0.7

        # For each batch in the dataloader
        for batch, data in enumerate(dataloader, 0):

            # Update D network: minimize -log(D(x)) - log(1-D(G(z))
            #################################################################################

            netD.zero_grad()

            # Form batch
            real_input = data[0].to(device)

            b_size = real_input.size(0) #current batch size
            label = torch.full((b_size,), real_label, device=device)

            # Forward pass real batch through D
            output = netD(real_input).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            errD_real.backward()
            avg_real_D = output.mean().item()

            # Generate a batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)

            # Generate fake images batch with G
            fake = netG(noise)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)

            # Calculate the gradients for this batch
            errD_fake.backward()
            avg_fake_D_before = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            optimizerD.step()

            #Update G network: maximize log(D(G(z)))
            #################################################################################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)

            # Calculate G's loss based on this output
            errG = criterion(output, label)

            # Calculate gradients for G
            errG.backward()
            avg_fake_D_after = output.mean().item()

            # Update G
            optimizerG.step()

            # Output training stats
            if (batch % print_every == 0) | (batch == len(dataloader)-1):
                print('[%d/%d][%d/%d]\t\t\t Loss_D: %.4f\t Loss_G: %.4f\t D output: real %.4f\t fake (before update | '
                      'after update): %.4f | %.4f'
                      % (epoch, num_epochs, batch, len(dataloader)-1,
                         errD.item(), errG.item(), avg_real_D, avg_fake_D_before, avg_fake_D_after))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1
        # save the model when it has the lowest Frechet Inception Distance
        if save_model:
            if (epoch % fid_every == 0) or (epoch == num_epochs-1):
                run_generator(model=netG, batch_size=2000, save_path=save_fake, RGB=True)
                fid = calculate_fid_given_paths(paths=[os.path.join(resized_RGB_dataroot, '1Les'), save_fake], batch_size=50, cuda=True, dims=2048)

                FIDs.append(fid)
                print('-' * 40 + 'FID {}'.format(fid))

                if(fid < min_fid):
                    torch.save(netG.state_dict(), save_fig_path + G_model_name)
                    min_fid = fid
                    print('model saved')

        cprint('-' * 40 + ' Epoch {} finished, average losses: Loss_D: {:.5f} Loss_G {:.5f}'.format(epoch, np.mean(D_losses[-len(dataloader):]), np.mean(G_losses[-len(dataloader):])),
               color='cyan',
               attrs=['bold'])

    #save the last model
    torch.save(netG.state_dict(), save_fig_path + G_model_name + '_last')

    #record the ending time
    end = time.time()
    cprint('Min FID = {}, Elapsed time for training {} Min'.format(min_fid, (end-start)/60), color='green', attrs=['bold'])

    # Results
    fig = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations"), plt.ylabel("Loss"), plt.legend()
    plt.savefig(os.path.join(save_fig_path, 'Losses_' + configuration), format='svg', dpi=1000)

    plt.figure()
    plt.title("Generator and Discriminator Average Loss During Training")

    Gmov_avg = np.convolve(G_losses, np.ones(30) / 30, mode= 'same')
    Dmov_avg = np.convolve(D_losses, np.ones(30) / 30, mode= 'same')

    seaborn.lineplot(np.arange(29, len(G_losses)-29), Gmov_avg[29: -29], label= 'G', dashes=True)
    seaborn.lineplot(np.arange(29, len(D_losses)-29), Dmov_avg[29: -29], label= 'D', dashes=True)

    plt.xlabel("iterations"), plt.ylabel("Average Loss"), plt.legend()
    plt.savefig(os.path.join(save_fig_path, 'AvgLosses_' + configuration), format='svg', dpi=1000)

    ######################################################################
    # **Visualization of G’s progression**

    ######################################################################
    # **Real Images vs. Fake Images**

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(os.path.join(save_fig_path, 'training_images' + configuration), format='svg', dpi=1000)

    # Plot the real images
    plt.figure(figsize=(8, 8))
    plt.axis("off"), plt.title("Fake Images")
    fake=  netG(fixed_noise).detach().cpu()/2 + 0.5
    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(os.path.join(save_fig_path, 'fake_images_' + configuration), format='svg', dpi=1000)

    # plot Frechet Inception Distances
    if save_model:
        plt.figure()
        plt.plot(FIDs)

        plt.xlabel('every {} epochs'.format(fid_every)), plt.ylabel('FID value')
        plt.title('Frechet Inception Distance')
        plt.grid()

        Fmov_avg = np.convolve(FIDs, np.ones(5) / 5, mode= 'same')
        plt.plot(np.arange(4, len(Fmov_avg) - 4), Fmov_avg[4: -4], label='average FID', dashes=[6,2])
        plt.legend()
        plt.savefig(os.path.join(save_fig_path, 'FID_' + configuration), format='svg', dpi=1000)

    plt.show()

    print('configuration:\n', configuration), print('Generator', G_model_name)

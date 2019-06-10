"""
Author: Basel Alyafi
Master Thesis Project
Erasmus Mundus Joint Master in Medical Imaging and Applications
"""
#How to Run
#Here We include an example of how to create a new instance, load pretrained weights and generate any number of mammographic lesinos.
from Gans.DCGAN import Generator
from helpers import run_generator

# the name of the generator
model_name = 'mass_calcification_gen'
n_imgs = 100
# the path where to save_model the images
imgs_path = '.../dst'
src_path = '.../'

# create an instance of the generator
model = Generator(ngpu=1, nz=200, ngf=45, nc=1)

# load the pretrained weights
model.load_state_dict(torch.load(src_path + model_name))

#run the trained generator to generate n_imgs images at imgs_path
run_generator(model=model, batch_size=n_imgs, save_path=imgs_path, RGB=False)

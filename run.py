import numpy as np
import torch
import torch.optim as optim
from train import *
from model import *
from util import *

batch_size = 32
img_size = 32

train_loader = get_dataloader(batch_size, img_size, 'processed_celeba_small/')

# Define model hyperparams
d_conv_dim = 32
g_conv_dim = 128
z_size = 100

D, G = build_network(d_conv_dim, g_conv_dim, z_size)

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')

# Create optimizers for the discriminator D and generator G

# params
lr = 0.0002
beta1=0.5
beta2=0.999 # default value


# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

n_epochs = 20

# call training function
losses = train(D, G, d_optimizer, g_optimizer, train_on_gpu,z_size, train_loader, n_epochs=n_epochs)
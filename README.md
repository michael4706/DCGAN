# DCGAN
This is the implementation of DCGAN
The implentation follows the instrcutions from [original DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) by Alec Radford & Luke Metz, Soumith Chintala

# Dataset
This network is trained on the image dataset [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). But feel free to use your own dataset to create a DCGAN!

# Background
DCGAN consists of a generator and a discriminator, where the two parts are both constructed by convolutional neural networks(CNN). The generator takes in a latent vector z, then construct a fake image through series of deconvolution. The discriminator takes in an image(real and fake), then assign a score to the image to indicate whether or not it thinks the image is real or fake.

# DCGAN Structure


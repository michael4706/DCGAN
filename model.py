import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, batch_norm = True):
    layers = []
    
    conv_1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                       stride = stride, padding = padding, bias=False)
    layers.append(conv_1)
    
    if batch_norm:
        batch_layer = nn.BatchNorm2d(num_features = out_channels)
        layers.append(batch_layer)
    
    return nn.Sequential(*layers)

class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()
        
        self.conv_dim = conv_dim
        
        #32x32x3 --> (32-4+2)/2+1 = 16 --> 16x16x32
        self.conv1 = conv_layer(in_channels = 3, out_channels = conv_dim, batch_norm = False)
        
        #16x16x64 --> (16-4+2)/2+1 = 8 --> 8x8x64
        self.conv2 = conv_layer(in_channels = conv_dim, out_channels = conv_dim * 2, batch_norm = True)
        
        #8x8x128 --> (8-4+2)/2+1 = 4 --> 4x4x128
        self.conv3 = conv_layer(in_channels =  conv_dim * 2, out_channels = conv_dim * 4, batch_norm = True)
        
        #4x4x256 --> (4-4+2)/2+1 = 2 --> 2x2x256
        self.conv4 = conv_layer(in_channels = conv_dim * 4, out_channels = conv_dim * 8, batch_norm = True)
        
        self.dense = nn.Linear(in_features = 2 * 2 * conv_dim * 8, out_features = 1)

        # complete init function
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(-1, 2 * 2 * self.conv_dim * 8)
        out = self.dense(x)
        
        
        return out

def deconv_layer(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, batch_norm = True):
    layers = []
    
    deconv = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, 
                                kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
    layers.append(deconv)
    
    if batch_norm:
        batch_layer = nn.BatchNorm2d(num_features = out_channels)
        layers.append(batch_layer)
    
    return nn.Sequential(*layers)

class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()
        
        self.conv_dim = conv_dim
        self.dense = nn.Linear(in_features = z_size, out_features = 4 * 4 * conv_dim * 4)
        
        #4x4x512 --> (4-1) * 2 - 2 * 1 + (4-1) + 1 = 8 --> 8x8x256
        self.deconv1 = deconv_layer(in_channels = conv_dim * 4, out_channels = conv_dim * 2)
        
        #8x8x256 --> (8-1) * 2 - 2*1 + (4-1) + 1 = 16x16x128
        self.deconv2 = deconv_layer(in_channels = conv_dim * 2, out_channels = conv_dim)
        
        #16x16x128 --> (16-1) * 2 - 2*1 + (4-1) + 1 = 32x32x3
        self.deconv3 = deconv_layer(in_channels = conv_dim, out_channels = 3)

        # complete init function
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.dense(x)
        x = x.view(-1,self.conv_dim * 4,4,4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.tanh(self.deconv3(x))
        
        return x

def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("Linear") != -1:
        m.weight.data.normal_(0, 0.02)
        
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    

def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)
    
    return D, G
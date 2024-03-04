import numpy as np
import torch.nn as nn

# Define linear generator network
class LinearGenerator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super().__init__()

        self.img_shape = img_shape

        def layer_block(input_size, output_size, normalize=True):
            layers = [nn.Linear(input_size, output_size)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_size, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *layer_block(latent_dim, 128, normalize=False),
			*layer_block(128, 256),
			*layer_block(256, 512),
			*layer_block(512, 1024),
			nn.Linear(1024, int(np.prod(img_shape))), 
			nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
    
class CNN_Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super().__init__()

        self.img_shape = img_shape

        def conv_block(in_channels, out_channels, kernel_size, stride, padding, normalize=True):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *conv_block(latent_dim, 1024, 4, 1, 0, normalize=False),
            *conv_block(1024, 512, 4, 2, 1),
            *conv_block(512, 256, 4, 2, 1),
            *conv_block(256, 128, 4, 2, 1),
            nn.ConvTranspose2d(128, img_shape[0], 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)  # Reshape input for convolutional layers
        img = self.model(z)
        return img
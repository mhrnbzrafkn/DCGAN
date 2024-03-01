import numpy as np
import torch.nn as nn

# Define linear generator network
class linear_Generator(nn.Module):
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
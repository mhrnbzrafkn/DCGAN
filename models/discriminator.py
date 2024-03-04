import numpy as np
import torch.nn as nn

# Define linear discriminator network
class LinearDiscriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
			nn.Linear(int(np.prod(img_shape)), 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        verdict = self.model(img_flat)
        return verdict

class CNN_Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.img_shape = img_shape

        def conv_block(in_channels, out_channels, kernel_size, stride, padding, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *conv_block(img_shape[0], 128, 4, 2, 1, normalize=False),
            *conv_block(128, 256, 4, 2, 1),
            *conv_block(256, 512, 4, 2, 1),
            *conv_block(512, 1024, 4, 2, 1),
            nn.Conv2d(1024, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)

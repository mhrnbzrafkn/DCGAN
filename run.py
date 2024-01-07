import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from gan_models import *

# Save models
def save_models(generator, discriminator, epoch, saving_outputs_step):
    # Define file paths for the models
    generator_path = f'models/generator-{epoch}.pth'
    discriminator_path = f'models/discriminator-{epoch}.pth'

    # Check if old models exist and remove them
    if epoch > 0 and os.path.exists(f'models/generator-{epoch - saving_outputs_step}.pth') and os.path.exists(f'models/discriminator-{epoch - saving_outputs_step}.pth'):
        os.remove(f'models/generator-{epoch - saving_outputs_step}.pth')
        os.remove(f'models/discriminator-{epoch - saving_outputs_step}.pth')

    # Save new models
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)

def visualize_outputs(losses, images_for_gif):
    # Visualizing the losses at every epoch
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Generator')
    plt.plot(losses.T[1], label='Discriminator')
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'outputs/loss_plot.png')

    # Creating a gif of generated images at every epoch
    imageio.mimwrite(f'outputs/generated_images.gif', images_for_gif, fps=20)

# Initializing the weights with small random values
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def train(losses, images_for_gif):
    # Initialize networks
    generator = linear_Generator(img_shape, latent_dim)
    generator.apply(weights_init)
    discriminator = linear_Discriminator(img_shape)
    discriminator.apply(weights_init)

    # Loss function and optimizers
    adversarial_loss  = nn.BCELoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=lr)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr)

    # Data preprocessing and loading
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))  # only one channel for grayscale images
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loading mnist dataset
    # dataset = datasets.MNIST(root="mnist_data", train=True, download=True, transform=transform)

    # Loading my custome dataset
    dataset = datasets.ImageFolder(root="training-images", transform=transform)

    # Generate data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    Tensor = torch.FloatTensor

    # Training loop
    for epoch in range(epochs):
        for i, (images, _) in enumerate(data_loader):
            # Configure input
            real_images = Variable(images.type(Tensor))

            # Adversarial ground truths
            real_output = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False)
            fake_output = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False)

            # Training Generator
            optimizer_generator.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], latent_dim))))
            generated_images = generator(z)
            generator_loss = adversarial_loss(discriminator(generated_images), real_output)
            generator_loss.backward()
            optimizer_generator.step()

            # Train discriminator
            optimizer_discriminator.zero_grad()
            discriminator_loss_real = adversarial_loss(discriminator(real_images), real_output)
            discriminator_loss_fake = adversarial_loss(discriminator(generated_images.detach()), fake_output)
            discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2
            discriminator_loss.backward()
            optimizer_discriminator.step()
            
            print(f"[Epoch {epoch:=4d}/{epochs}] [Batch {i:=4d}/{len(data_loader)}] ---> "
                f"[D Loss: {discriminator_loss.item():.6f}] [G Loss: {generator_loss.item():.6f}]")

        if epoch % saving_outputs_step == 0:
            losses.append((generator_loss.item(), discriminator_loss.item()))
            image_filename = f'generated_images/{epoch}.png'
            generator_inputs = Variable(Tensor(np.random.normal(0, 1, (16, latent_dim))))
            generator_outputs = generator(generator_inputs)
            save_image(generator_outputs.data, image_filename, nrow=4, normalize=True)
            images_for_gif.append(imageio.v2.imread(image_filename))

            # Save models
            save_models(generator, discriminator, epoch, saving_outputs_step)

    return losses, images_for_gif

if __name__ == "__main__":
    # Create a folder to save generated images
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Hyperparameters
    latent_dim = 32
    img_shape = (3, 64, 64)
    batch_size = 1024
    epochs = 20000
    lr = 0.0001
    saving_outputs_step = 10

    # Visual data
    losses = []
    images_for_gif = []

    try:
        losses, images_for_gif = train(losses, images_for_gif)

    except KeyboardInterrupt:
        visualize_outputs(losses, images_for_gif)

    finally:
        print("Training interrupted by user.")
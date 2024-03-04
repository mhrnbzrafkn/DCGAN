import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from settings import *
from utils import *
from models.discriminator import *
from models.generator import *

def train_loop(losses, images_for_gif):
    print("Generating models. (Generator and Discriminator)")
    # Initialize networks
    generator = LinearGenerator(OUTPUT_IMAGE_SHAPE, INPUT_VECTOR_LENGTH)
    generator.apply(weights_init)
    discriminator = LinearDiscriminator(OUTPUT_IMAGE_SHAPE)
    discriminator.apply(weights_init)

    print("Setting the optimizers...")
    # Loss function and optimizers
    adversarial_loss  = nn.BCELoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # Data preprocessing and loading
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))  # only one channel for grayscale images
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Loading custome dataset.")
    # Loading my custome dataset
    dataset = datasets.ImageFolder(root=TRAINING_IMAGES_PATH, transform=transform)

    # Define the sizes of training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print("Generate data loaders.")
    # Generate data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    Tensor = torch.FloatTensor

    print("Training loop started...")
    # Training loop
    for epoch in range(EPOCHS):
        # Training
        generator.train()
        discriminator.train()
        for i, (images, _) in enumerate(train_loader):
            # Configure input
            real_images = Variable(images.type(Tensor))

            # Adversarial ground truths
            real_output = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False)
            fake_output = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False)

            # Training Generator
            optimizer_generator.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], INPUT_VECTOR_LENGTH))))
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
            
            print(f"[Epoch {epoch + 1:=4d}/{EPOCHS}] [Batch {i + 1:=4d}/{len(train_loader)}] ---> "
                f"[D Loss: {discriminator_loss.item():.6f}] [G Loss: {generator_loss.item():.6f}]")

        # Validation
        generator.eval()
        discriminator.eval()
        val_losses = []
        with torch.no_grad():
            for images, _ in val_loader:
                real_images = Variable(images.type(Tensor))
                real_output = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False)
                fake_output = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False)

                z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], INPUT_VECTOR_LENGTH))))
                generated_images = generator(z)
                generator_loss = adversarial_loss(discriminator(generated_images), real_output)

                discriminator_loss_real = adversarial_loss(discriminator(real_images), real_output)
                discriminator_loss_fake = adversarial_loss(discriminator(generated_images.detach()), fake_output)
                discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2

                val_losses.append((generator_loss.item(), discriminator_loss.item()))

        avg_val_generator_loss = sum([loss[0] for loss in val_losses]) / len(val_losses)
        avg_val_discriminator_loss = sum([loss[1] for loss in val_losses]) / len(val_losses)
        print(f"Validation Loss: [D Loss: {avg_val_discriminator_loss:.6f}] [G Loss: {avg_val_generator_loss:.6f}]")

        # Save losses
        losses.append((avg_val_generator_loss, avg_val_discriminator_loss))

        if epoch % SAVE_OUTPUT_IMAGE_STEP == 0:
            image_filename = f'{GENERATED_IMAGES_PATH}/{epoch}.png'
            generator_inputs = Variable(Tensor(np.random.normal(0, 1, (16, INPUT_VECTOR_LENGTH))))
            generator_outputs = generator(generator_inputs)
            save_image(generator_outputs.data, image_filename, nrow=4, normalize=True)
            images_for_gif.append(imageio.v2.imread(image_filename))

            # Save models
            save_models(generator, discriminator, epoch, SAVE_OUTPUT_IMAGE_STEP)

    return losses, images_for_gif

if __name__ == "__main__":
    # Visualization parameters
    losses = []
    output_gif_images = []

    try:
        losses, output_gif_images = train_loop(losses, output_gif_images)

    except KeyboardInterrupt:
        report_training_process(losses, output_gif_images)

    finally:
        print("Training interrupted by user.")
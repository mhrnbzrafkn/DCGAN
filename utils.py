import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from settings import *

# Save models
def save_models(generator, discriminator, epoch, saving_outputs_step):
    # Define file paths for the models
    generator_path = f'{TRAINED_MODELS_PATH}/generator-{epoch}.pth'
    discriminator_path = f'{TRAINED_MODELS_PATH}/discriminator-{epoch}.pth'

    # Check if old models exist and remove them
    if epoch > 0 and os.path.exists(f'{TRAINED_MODELS_PATH}/generator-{epoch - saving_outputs_step}.pth') and os.path.exists(f'{TRAINED_MODELS_PATH}/discriminator-{epoch - saving_outputs_step}.pth'):
        os.remove(f'{TRAINED_MODELS_PATH}/generator-{epoch - saving_outputs_step}.pth')
        os.remove(f'{TRAINED_MODELS_PATH}/discriminator-{epoch - saving_outputs_step}.pth')

    # Save new models
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)

def report_training_process(losses, images_for_gif):
    # Visualizing the losses at every epoch
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Generator')
    plt.plot(losses.T[1], label='Discriminator')
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{OUTPUT_REPORT_PATH}/loss_plot.png')

    # Creating a gif of generated images at every epoch
    imageio.mimwrite(f'{OUTPUT_REPORT_PATH}/generated_images.gif', images_for_gif, fps=20)

# Initializing the weights with small random values
def weights_init(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        nn.init.normal_(model.weight.data, 0.0, 0.02)
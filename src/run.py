import os
import re
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

### default settings ###
RESOURCES_PATH = './training-src'
BASE_PATH = f'{RESOURCES_PATH}/output_results'
# create base folder
os.makedirs(BASE_PATH, exist_ok=True)

# create a folder to save images that created while training model
GENERATED_IMAGES_PATH = f'{BASE_PATH}/generated_images'
os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)

# create a folder to save models after training process
TRAINED_MODELS_PATH = f'{BASE_PATH}/trained_models'
os.makedirs(TRAINED_MODELS_PATH, exist_ok=True)

# create a folder to save final report of training process
OUTPUT_REPORT_PATH = f'{BASE_PATH}/output_report'
os.makedirs(OUTPUT_REPORT_PATH, exist_ok=True)

# create a folder to save generated images using trained model
GENERATED_TEST_IMAGES_PATH = f'{BASE_PATH}/generated_test_images'
os.makedirs(GENERATED_TEST_IMAGES_PATH, exist_ok=True)

TRAINING_IMAGES_PATH = f'{RESOURCES_PATH}/training_images'

### Hyperparameters ###
input_vector_length = 16
neuralnet_deep = 64
image_shape = (3, 64, 64)

max_epochs = 500
batch_size = 16
save_output_image_step = 1

momentum = 0.6
eps = 0.00001

discriminator_learning_rate = 0.0002
generator_learning_rate = discriminator_learning_rate/4

scheduler_step = 10
scheduler_gamma = 0.8

class Generator(nn.Module):
    def __init__(self, noise_size: int, deep: int):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: [64]
            nn.Linear(noise_size, deep*8*4*4),
            nn.ReLU(),
            # [8192]

            nn.Unflatten(1, (deep*8, 4, 4)),
            # [512, 4, 4]

            nn.ConvTranspose2d(deep*8, deep*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(deep*4, eps, momentum),
            nn.ReLU(),
            # [256, 8, 8]

            nn.ConvTranspose2d(deep*4, deep*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(deep*2, eps, momentum),
            nn.ReLU(),
            # [128, 16, 16]

            nn.Upsample(scale_factor=2),
            # [128, 32, 32]

            nn.ConvTranspose2d(deep*2, deep, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(deep, eps, momentum),
            nn.ReLU(),
            # [64, 32, 32]

            nn.ConvTranspose2d(deep, int(deep/2), kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(int(deep/2), eps, momentum),
            nn.ReLU(),
            # [32, 64, 64]

            nn.ConvTranspose2d(int(deep/2), int(deep/4), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(deep/4), eps, momentum),
            nn.ReLU(),
            # [16, 64, 64]

            nn.Upsample(scale_factor=2),
            # [16, 128, 128]

            nn.Conv2d(int(deep/4), image_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            # Output: [3, 64, 64]
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple, deep: int):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: [3, 64, 64]
            nn.Conv2d(img_shape[0], deep, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(deep, eps, momentum),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            # [64, 32, 32]

            nn.Conv2d(deep, deep*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(deep*2, eps, momentum),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            # [128, 16, 16]

            nn.Conv2d(deep*2, deep*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(deep*4, eps, momentum),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            # [256, 8, 8]

            nn.Conv2d(deep*4, deep*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(deep*8, eps, momentum),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            # [512, 4, 4]

            nn.Conv2d(deep*8, deep*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(deep*8, eps, momentum),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            # [512, 4, 4]
            
            nn.Flatten(),
            # [8192]

            nn.Linear(deep*8*4*4, 1),
            nn.Sigmoid()
            # Output: [1]
        )

    def forward(self, x):
        return self.model(x)

### Save models ###
def save_models(generator: Generator, discriminator: Discriminator, epoch: int, saving_outputs_step: int):
    # Define file paths for the models
    generator_path = f'{TRAINED_MODELS_PATH}/generator-{epoch+1}.pt'
    generator_scripted_path = f'{TRAINED_MODELS_PATH}/generator-scripted-{epoch+1}.pt'

    discriminator_path = f'{TRAINED_MODELS_PATH}/discriminator-{epoch+1}.pt'
    discriminator_scripted_path = f'{TRAINED_MODELS_PATH}/discriminator-scripted-{epoch+1}.pt'

    # Save new models
    torch.save(generator.state_dict(), generator_path)
    scripted_generator = torch.jit.script(generator)
    scripted_generator.save(generator_scripted_path)

    torch.save(discriminator.state_dict(), discriminator_path)
    scripted_discriminator = torch.jit.script(discriminator)
    scripted_discriminator.save(discriminator_scripted_path)

    # Check if old models exist and remove them
    if epoch >= saving_outputs_step:
        prev_epoch = epoch - saving_outputs_step + 1
        prev_generator_path = f'{TRAINED_MODELS_PATH}/generator-{prev_epoch}.pt'
        prev_scripted_generator_path = f'{TRAINED_MODELS_PATH}/generator-scripted-{prev_epoch}.pt'

        prev_discriminator_path = f'{TRAINED_MODELS_PATH}/discriminator-{prev_epoch}.pt'
        prev_scripted_discriminator_path = f'{TRAINED_MODELS_PATH}/discriminator-scripted-{prev_epoch}.pt'

        if os.path.exists(prev_generator_path):
            os.remove(prev_generator_path)
        if os.path.exists(prev_scripted_generator_path):
            os.remove(prev_scripted_generator_path)
        
        if os.path.exists(prev_discriminator_path):
            os.remove(prev_discriminator_path)
        if os.path.exists(prev_scripted_discriminator_path):
            os.remove(prev_scripted_discriminator_path)

def load_scripted_model(model_path: str):
    """
    Load scripted torch model.

    Args:
    - model_path (str): Path to the scripted torch model.
    """
    return torch.jit.load(model_path)

def load_model(model: nn.Module, model_path: str):
    """
    Load torch model.

    Args:
    - model (torch.nn.Module): Defined model class.
    - model_path (str): Path to the torch model.
    """
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def load_models(device: torch.device):
    generator_pattern = re.compile(r'^generator-\d+\.pt$')
    all_files = os.listdir(TRAINED_MODELS_PATH)
    matching_files = [file for file in all_files if generator_pattern.match(file)]
    if matching_files:
        print("01- Loading Generator.")
        generator = Generator(input_vector_length , neuralnet_deep).to(device)
        matching_file_path = os.path.join(TRAINED_MODELS_PATH, matching_files[0])
        generator_state_dict = torch.load(matching_file_path, map_location=device)
        generator.load_state_dict(generator_state_dict)
        generator.train()
    else:
        print("01- Creating Generator.")
        generator = Generator(input_vector_length , neuralnet_deep).to(device)

    discriminator_pattern = re.compile(r'^discriminator-\d+\.pt$')
    all_files = os.listdir(TRAINED_MODELS_PATH)
    matching_files = [file for file in all_files if discriminator_pattern.match(file)]
    if matching_files:
        print("01- Loading Discriminator.")
        discriminator = Discriminator(image_shape, neuralnet_deep).to(device)
        matching_file_path = os.path.join(TRAINED_MODELS_PATH, matching_files[0])
        discriminator_state_dict = torch.load(matching_file_path, map_location=device)
        discriminator.load_state_dict(discriminator_state_dict)
        discriminator.train()
    else:
        print("02- Creating Discriminator.")
        discriminator = Discriminator(image_shape, neuralnet_deep).to(device)

    return generator, discriminator

def save_losses_to_csv(generator_losses: list, discriminator_losses: list, generator_csv_path: str, discriminator_csv_path: str):
    """
    Save generator and discriminator losses to separate CSV files.

    Args:
    - generator_losses (list): List of generator losses.
    - discriminator_losses (list): List of discriminator losses.
    - generator_csv_path (str): File path for the CSV file to save generator losses.
    - discriminator_csv_path (str): File path for the CSV file to save discriminator losses.
    """
    # Write generator losses to CSV file
    with open(generator_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Generator Loss'])
        for epoch, loss in enumerate(generator_losses, start=1):
            writer.writerow([epoch, loss])

    # Write discriminator losses to CSV file
    with open(discriminator_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Discriminator Loss'])
        for epoch, loss in enumerate(discriminator_losses, start=1):
            writer.writerow([epoch, loss])

### Data preprocessing and loading ###
def create_transform():
    return transforms.Compose([
        transforms.Resize((image_shape[1], image_shape[2])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# def train(losses, output_gif_images):
def train(generator_losses: list, discriminator_losses: list):
    # Check if GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'00- Using {device}.')

    generator, discriminator = load_models(device)

    print("03- Setting the optimizers completed.")
    adversarial_loss  = nn.BCELoss()

    generator_optimizer = optim.Adam(generator.parameters(), lr=generator_learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
    generator_scheduler = StepLR(generator_optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=discriminator_learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
    discriminator_scheduler = StepLR(discriminator_optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # Data preprocessing and loading
    transform = create_transform()

    print("04- Loading custome dataset.")
    # Loading my custome dataset
    dataset = datasets.ImageFolder(root=TRAINING_IMAGES_PATH, transform=transform)

    print("05- Generate data loaders.")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    Tensor = torch.FloatTensor

    # generate random vectors
    # static_vectors = Variable(Tensor(np.random.normal(0, 1, (len(dataset), input_vector_length ))).float()).to(device)

    print("06- Training loop started.")
    # Training loop
    for current_epoch in range(max_epochs):
        # Training
        generator.train()
        discriminator.train()

        generator_losses_in_epoch = []
        discriminator_losses_in_epoch = []

        for i, (images, _) in enumerate(data_loader):
            # Configure input
            real_images = Variable(images.type(Tensor)).to(device)

            # Adversarial ground truths
            real_output = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False).to(device)
            fake_output = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False).to(device)

            # Training Generator
            generator_optimizer.zero_grad()
            # Use random vectors
            z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], input_vector_length ))).float()).to(device)
            # Use static vectors
            # z = static_vectors[i * batch_size : (i + 1) * batch_size]
            generated_images = generator(z).to(device)
            generator_loss = adversarial_loss(discriminator(generated_images), real_output)
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            generator_optimizer.step()

            # Train discriminator
            discriminator_optimizer.zero_grad()
            discriminator_loss_real = adversarial_loss(discriminator(real_images), real_output)
            discriminator_loss_fake = adversarial_loss(discriminator(generated_images.detach()), fake_output)
            discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            discriminator_optimizer.step()

            generator_losses_in_epoch.append(generator_loss.item())
            discriminator_losses_in_epoch.append(discriminator_loss.item())

            print(f"[Epoch {current_epoch + 1:=4d}/{max_epochs}] [Batch {i + 1:=4d}/{len(data_loader)}] ---> "
                f"[D Loss: {discriminator_loss.item():.6f}] [G Loss: {generator_loss.item():.6f}]")
        
        generator_scheduler.step()
        discriminator_scheduler.step()

        # Save losses
        generator_losses.append(sum(generator_losses_in_epoch)/len(generator_losses_in_epoch))
        discriminator_losses.append(sum(discriminator_losses_in_epoch)/len(discriminator_losses_in_epoch))

        if current_epoch % save_output_image_step == 0:
            image_filename = f'{GENERATED_IMAGES_PATH}/{current_epoch+1}.png'
            generator_inputs = Variable(Tensor(np.random.normal(0, 1, (16, input_vector_length )))).to(device)
            generator_outputs = generator(generator_inputs).to(device)
            save_image(generator_outputs.data, image_filename, nrow=4, normalize=True)

            # Save models
            save_models(generator, discriminator, current_epoch, save_output_image_step)

            save_losses_to_csv(generator_losses,
                               discriminator_losses,
                               f'{OUTPUT_REPORT_PATH}/generator_losses.csv',
                               f'{OUTPUT_REPORT_PATH}/discriminator_losses.csv')

    # return losses, output_gif_images
    return generator_losses, discriminator_losses

### Visualization parameters ###
generator_losses = []
discriminator_losses = []

generator_losses, discriminator_losses, train(generator_losses, discriminator_losses)
save_losses_to_csv(generator_losses,
                   discriminator_losses,
                   f'{OUTPUT_REPORT_PATH}/generator_losses.csv',
                   f'{OUTPUT_REPORT_PATH}/discriminator_losses.csv')
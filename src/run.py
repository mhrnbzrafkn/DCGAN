import os
import re
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
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
input_noise_size = 128
neural_net_deep = 64
image_shape = (3, 64, 64)
max_epochs = 200
batch_size = 16
save_output_image_step = 1
discriminator_learning_rate = 0.0001
generator_learning_rate = 0.0001
# generator_learning_rate = discriminator_learning_rate/2

class Generator(nn.Module):
    def __init__(self, noise_size: int, deep: int):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: [128]
            nn.ConvTranspose2d(noise_size, deep*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(deep*8, momentum=0.8),
            # [512, 4, 4]

            nn.ConvTranspose2d(deep*8, deep*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(deep*4, momentum=0.8),
            # [256, 8, 8]

            nn.ConvTranspose2d(deep*4, deep*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(deep*2, momentum=0.8),
            # [128, 16, 16]

            nn.ConvTranspose2d(deep*2, deep, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(deep, momentum=0.8),
            # [64, 32, 32]

            nn.ConvTranspose2d(deep, image_shape[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # [3, 64, 64]
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple, deep: int):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: [3, 64, 64]
            nn.Conv2d(img_shape[0], deep, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # [64, 32, 32]

            nn.Conv2d(deep, deep*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(deep*2, momentum=0.8),
            nn.Dropout2d(0.2),
            # [128, 16, 16]

            nn.Conv2d(deep*2, deep*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(deep*4, momentum=0.8),
            nn.Dropout2d(0.2),
            # [256, 8, 8]

            nn.Conv2d(deep*4, deep*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(deep*8, momentum=0.8),
            nn.Dropout2d(0.2),
            # [512, 4, 4]

            nn.Conv2d(deep*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
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
        print("02- Loading Generator.")
        generator = Generator(input_noise_size , neural_net_deep).to(device)
        matching_file_path = os.path.join(TRAINED_MODELS_PATH, matching_files[0])
        generator_state_dict = torch.load(matching_file_path, map_location=device)
        generator.load_state_dict(generator_state_dict)
        generator.train()
    else:
        print("02- Creating Generator.")
        generator = Generator(input_noise_size , neural_net_deep).to(device)

    discriminator_pattern = re.compile(r'^discriminator-\d+\.pt$')
    all_files = os.listdir(TRAINED_MODELS_PATH)
    matching_files = [file for file in all_files if discriminator_pattern.match(file)]
    if matching_files:
        print("02- Loading Discriminator.")
        discriminator = Discriminator(image_shape, neural_net_deep).to(device)
        matching_file_path = os.path.join(TRAINED_MODELS_PATH, matching_files[0])
        discriminator_state_dict = torch.load(matching_file_path, map_location=device)
        discriminator.load_state_dict(discriminator_state_dict)
        discriminator.train()
    else:
        print("02- Creating Discriminator.")
        discriminator = Discriminator(image_shape, neural_net_deep).to(device)

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

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

### Data preprocessing and loading ###
def create_transform():
    return transforms.Compose([
        transforms.Resize((image_shape[1], image_shape[2])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# def train(losses, output_gif_images):
def train_model(generator_losses: list, discriminator_losses: list):
    # Check if GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'01- Using {device} to train model.')

    generator, discriminator = load_models(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    print('02- Models were built.')

    adversarial_loss  = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=generator_learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=discriminator_learning_rate, betas=(0.5, 0.999))
    print("03- Setting the optimizers completed.")

    print("04- Loading custome dataset.")
    # Data preprocessing and loading
    transform = create_transform()
    # Loading my custome dataset
    dataset = datasets.ImageFolder(root=TRAINING_IMAGES_PATH, transform=transform)

    print("05- Generate data loaders.")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # generate vectors
    fixed_noise = torch.randn(16, input_noise_size, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    print("06- Training loop started.")
    # Training loop
    for current_epoch in range(max_epochs):
        # Training
        generator.train()
        discriminator.train()

        generator_losses_in_epoch = []
        discriminator_losses_in_epoch = []

        # for i, (images, _) in enumerate(data_loader):
        for i, data in enumerate(data_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            ## Train with all-real batch
            discriminator.zero_grad()
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size, ), real_label, dtype=torch.float, device=device)
            output = discriminator(real).view(-1)
            discriminator_real_error = adversarial_loss(output, label)
            discriminator_real_error.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, input_noise_size, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            discriminator_fake_error = adversarial_loss(output, label)
            discriminator_fake_error.backward()
            D_G_z1 = output.mean().item()
            discriminator_error = discriminator_real_error + discriminator_fake_error
            discriminator_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            generator_error = adversarial_loss(output, label)
            generator_error.backward()
            D_G_z2 = output.mean().item()
            generator_optimizer.step()

            generator_losses_in_epoch.append(generator_error.item())
            discriminator_losses_in_epoch.append(discriminator_error.item())

            # Logging
            max_epoch_length = len(str(max_epochs))
            max_batch_length = len(str(len(data_loader)))
            max_loss_length = 8
            max_d_length = 8
            print(f'[{current_epoch+1:0{max_epoch_length}d}/{max_epochs}]'
                  f'[{i+1:0{max_batch_length}d}/{len(data_loader)}] '
                  f'Loss_D: {discriminator_error.item():<{max_loss_length}.4f} '
                  f'Loss_G: {generator_error.item():<{max_loss_length}.4f} '
                  f'D(x): {D_x:<{max_d_length}.4f} '
                  f'D(G(z)): {D_G_z1:<{max_d_length}.4f} / {D_G_z2:<{max_d_length}.4f}')

        
        # # Save losses
        generator_losses.append(sum(generator_losses_in_epoch)/len(generator_losses_in_epoch))
        discriminator_losses.append(sum(discriminator_losses_in_epoch)/len(discriminator_losses_in_epoch))

        if current_epoch % save_output_image_step == 0:
            image_filename = f'{GENERATED_IMAGES_PATH}/{current_epoch+1}.png'
            generator_outputs = generator(fixed_noise.to(device))
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

generator_losses, discriminator_losses, train_model(generator_losses, discriminator_losses)
save_losses_to_csv(generator_losses,
                   discriminator_losses,
                   f'{OUTPUT_REPORT_PATH}/generator_losses.csv',
                   f'{OUTPUT_REPORT_PATH}/discriminator_losses.csv')
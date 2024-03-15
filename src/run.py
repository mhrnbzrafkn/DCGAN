import os
import torch
import imageio
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
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
INPUT_VECTOR_LENGTH = 64
OUTPUT_IMAGE_SHAPE = (3, 64, 64)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
SAVE_OUTPUT_IMAGE_STEP = 1

NEURALNET_DEEP = 64

### Generator Model ###
class Generator(nn.Module):
    def __init__(self, input_vector_length):
        super(Generator, self).__init__()
        self.deep = NEURALNET_DEEP

        self.fc1 = nn.Linear(input_vector_length, 4*4*self.deep*8)
        
        self.conv1 = nn.ConvTranspose2d(in_channels=self.deep*8, out_channels=self.deep*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.deep*4)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.ConvTranspose2d(in_channels=self.deep*4, out_channels=self.deep*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.deep*2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.ConvTranspose2d(in_channels=self.deep*2, out_channels=self.deep, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.deep)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.ConvTranspose2d(in_channels=self.deep, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.deep*8, 4, 4)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.tanh(x)
        return x

### Discriminator Model ###
class Discriminator(nn.Module):
    def __init__(self, output_img_shape):
        super(Discriminator, self).__init__()
        self.deep = NEURALNET_DEEP

        self.conv1 = nn.Conv2d(in_channels=output_img_shape[0], out_channels=self.deep*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.deep*8)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(in_channels=self.deep*8, out_channels=self.deep*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.deep*4)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(in_channels=self.deep*4, out_channels=self.deep*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.deep*2)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(in_channels=self.deep*2, out_channels=self.deep, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.deep)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(in_channels=self.deep, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.leaky_relu1(self.bn1(self.conv1(img)))
        x = self.leaky_relu2(self.bn2(self.conv2(x)))
        x = self.leaky_relu3(self.bn3(self.conv3(x)))
        x = self.leaky_relu4(self.bn4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        return x.view(-1, 1)

### Save models ###
def save_models(generator, discriminator, epoch, saving_outputs_step):
    # Define file paths for the models
    generator_path = f'{TRAINED_MODELS_PATH}/generator-{epoch+1}.pth'
    discriminator_path = f'{TRAINED_MODELS_PATH}/discriminator-{epoch+1}.pth'

    # Check if old models exist and remove them
    if (epoch > 0 and 
        os.path.exists(f'{TRAINED_MODELS_PATH}/generator-{epoch - saving_outputs_step}.pth') and
        os.path.exists(f'{TRAINED_MODELS_PATH}/discriminator-{epoch - saving_outputs_step}.pth')):
        os.remove(f'{TRAINED_MODELS_PATH}/generator-{epoch - saving_outputs_step}.pth')
        os.remove(f'{TRAINED_MODELS_PATH}/discriminator-{epoch - saving_outputs_step}.pth')

    # Save new models
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)

### Visualizing the losses at every epoch ###
def report_training_process(losses, images_for_gif):
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

### Data preprocessing and loading ###
def create_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def train(losses, images_for_gif):
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
        print("00- Using GPU for training.")
    else:
        device = torch.device("cpu")   # Use CPU
        print("00- GPU not available, using CPU for training.")

    print("01- Creating Generator completed.")
    generator = Generator(INPUT_VECTOR_LENGTH).to(device)
    # generator.apply(weights_init)

    print("02- Creating Discriminator completed.")
    discriminator = Discriminator(OUTPUT_IMAGE_SHAPE).to(device)
    # discriminator.apply(weights_init)

    print("03- Setting the optimizers completed.")
    adversarial_loss  = nn.BCELoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Data preprocessing and loading
    transform = create_transform()

    print("04- Loading custome dataset.")
    # Loading my custome dataset
    dataset = datasets.ImageFolder(root=TRAINING_IMAGES_PATH, transform=transform)

    # Define the sizes of training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print("05- Generate data loaders.")
    # Generate data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    Tensor = torch.FloatTensor

    print("06- Training loop started.")
    # Training loop
    for epoch in range(EPOCHS):
        # Training
        generator.train()
        discriminator.train()
        for i, (images, _) in enumerate(train_loader):
            # Configure input
            real_images = Variable(images.type(Tensor)).to(device)

            # Adversarial ground truths
            real_output = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False).to(device)
            fake_output = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False).to(device)

            # Training Generator
            optimizer_generator.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], INPUT_VECTOR_LENGTH))).float()).to(device)
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
            
            # Save losses
            # losses.append((generator_loss, discriminator_loss))
            # losses.append((generator_loss.detach().item(), discriminator_loss.detach().item()))

            print(f"[Epoch {epoch + 1:=4d}/{EPOCHS}] [Batch {i + 1:=4d}/{len(train_loader)}] ---> "
                f"[D Loss: {discriminator_loss.item():.6f}] [G Loss: {generator_loss.item():.6f}]")

        # Validation
        generator.eval()
        discriminator.eval()
        val_losses = []
        with torch.no_grad():
            for images, _ in val_loader:
                real_images = Variable(images.type(Tensor)).to(device)
                real_output = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False).to(device)
                fake_output = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False).to(device)

                z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], INPUT_VECTOR_LENGTH))).float()).to(device)
                generated_images = generator(z)
                generator_loss = adversarial_loss(discriminator(generated_images), real_output)

                discriminator_loss_real = adversarial_loss(discriminator(real_images), real_output)
                discriminator_loss_fake = adversarial_loss(discriminator(generated_images.detach()), fake_output)
                discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2

                val_losses.append((generator_loss.item(), discriminator_loss.item()))

        avg_val_generator_loss = sum([loss[0] for loss in val_losses]) / len(val_losses)
        avg_val_discriminator_loss = sum([loss[1] for loss in val_losses]) / len(val_losses)
        print(f"[Epoch {epoch + 1:=4d}/{EPOCHS}] [Batch {i + 1:=4d}/{len(train_loader)}] ---> "
                f"[D Loss: {avg_val_discriminator_loss:.6f}] [G Loss: {avg_val_generator_loss:.6f}]-(Validation report)")

        # Save losses
        losses.append((avg_val_generator_loss, avg_val_discriminator_loss))

        if epoch % SAVE_OUTPUT_IMAGE_STEP == 0:
            image_filename = f'{GENERATED_IMAGES_PATH}/{epoch+1}.png'
            generator_inputs = Variable(Tensor(np.random.normal(0, 1, (16, INPUT_VECTOR_LENGTH)))).to(device)
            generator_outputs = generator(generator_inputs)
            save_image(generator_outputs.data, image_filename, nrow=4, normalize=True)
            images_for_gif.append(imageio.v2.imread(image_filename))

            # Save models
            save_models(generator, discriminator, epoch, SAVE_OUTPUT_IMAGE_STEP)

    return losses, images_for_gif

### Visualization parameters ###
losses = []
output_gif_images = []

try:
    losses, output_gif_images = train(losses, output_gif_images)
except KeyboardInterrupt:
    report_training_process(losses, output_gif_images)
finally:
    print("Training interrupted by user.")
import torch
from run import *
import numpy as np

base_directory = './models'

# Instantiate the Generator class
generator = linear_Generator((3, 64, 64), 100)

# Load the state dictionary
state_dict = torch.load(f'{base_directory}/generator-900.pth', map_location=torch.device('cpu'))

# Load the state dictionary into the model
generator.load_state_dict(state_dict)

# Set the model to evaluation mode (important for models with Batch Normalization)
generator.eval()

image_filename = f'{base_directory}/output.png'

Tensor = torch.FloatTensor
generator_inputs = Variable(Tensor(np.random.normal(0, 1, (10000, 100))))
generator_outputs = generator(generator_inputs)
save_image(generator_outputs.data, image_filename, nrow=100, normalize=True)
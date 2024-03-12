import torch
import re
import math
from run import *

# to test the trained model, you can make a simple application like this code down below or make an API.

NUMBER_OF_OUTPUT_IMAGES = 1
OUTPUT_IMAGE_ROWS = int(math.sqrt(NUMBER_OF_OUTPUT_IMAGES))

# Define the pattern to match filenames
pattern = re.compile(r'^generator-\d+\.pth$')

# Get a list of all files in the directory
all_files = os.listdir(TRAINED_MODELS_PATH)

# Filter out files that match the pattern
matching_files = [file for file in all_files if pattern.match(file)]

# Load the state_dict from the first matching file
if matching_files:
    # Instantiate the Generator class
    generator = Generator(INPUT_VECTOR_LENGTH)
    # Load the state dictionary
    matching_file_path = os.path.join(TRAINED_MODELS_PATH, matching_files[0])
    state_dict = torch.load(matching_file_path, map_location=torch.device('cpu'))

    # Load the state dictionary into the model
    generator.load_state_dict(state_dict)
    # Set the model to evaluation mode (important for models with Batch Normalization)
    generator.eval()

    image_filename = f'{GENERATED_TEST_IMAGES_PATH}/output.png'
    Tensor = torch.FloatTensor
    inputs = [
        [-7, -6, 5, -3, -2, -1, -0.5, 0, 0.5, 0.7, 1, 2, 3, 5, 6, 8]
    ]
    # generator_inputs = Variable(Tensor(np.random.normal(0, 1, (NUMBER_OF_OUTPUT_IMAGES, INPUT_VECTOR_LENGTH))))
    generator_inputs = Variable(Tensor(inputs))
    generator_outputs = generator(generator_inputs)
    save_image(generator_outputs.data, image_filename, nrow=OUTPUT_IMAGE_ROWS, normalize=True)
else:
    print("No matching files found.")
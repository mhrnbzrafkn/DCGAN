import io
import re
import os
import math
import torch
import base64
import torchvision
import torch.nn as nn
from flask_cors import CORS
from torchvision.utils import save_image
from flask import Flask, request, jsonify, send_file

NUMBER_OF_OUTPUT_IMAGES = 1
OUTPUT_IMAGE_ROWS = int(math.sqrt(NUMBER_OF_OUTPUT_IMAGES))

BASE_PATH = './training-src/output_results'

INPUT_VECTOR_LENGTH = 8
TRAINED_MODELS_PATH = f'{BASE_PATH}/trained_models'

OUTPUT_IMAGE_FORMAT = 'JPEG'

### Generator Model ###
class Generator(nn.Module):
    def __init__(self, input_vector_length):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_vector_length, 4*4*1024)
        
        self.conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.tanh(x)
        return x

def generate_image(inputs, matching_files):
    # Instantiate the Generator class
    generator = Generator(INPUT_VECTOR_LENGTH)
    # Load the state dictionary
    matching_file_path = os.path.join(TRAINED_MODELS_PATH, matching_files[0])
    state_dict = torch.load(matching_file_path, map_location=torch.device('cpu'))

    # Load the state dictionary into the model
    generator.load_state_dict(state_dict)
    # Set the model to evaluation mode (important for models with Batch Normalization)
    generator.eval()

    # Generate image
    generator_inputs = torch.FloatTensor([inputs])
    generator_outputs = generator(generator_inputs)

    return generator_outputs

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the pattern to match filenames
pattern = re.compile(r'^generator-\d+\.pth$')

@app.route('/get-image-json', methods=['POST'])
def get_image_json():
    data = request.json
    inputs = data['inputs']
    
    # Get a list of all files in the directory
    all_files = os.listdir(TRAINED_MODELS_PATH)

    # Filter out files that match the pattern
    matching_files = [file for file in all_files if pattern.match(file)]

    # Load the state_dict from the first matching file
    if matching_files:
        generator_output = generate_image(inputs, matching_files)

        # Convert tensor to PIL Image
        pil_image = torchvision.transforms.ToPILImage()(generator_output[0].detach().cpu())

        # Convert PIL Image to bytes
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format=OUTPUT_IMAGE_FORMAT)
        image_bytes.seek(0)

        # Convert bytes to base64 encoded string
        base64_img_str = base64.b64encode(image_bytes.read()).decode()

        return jsonify({'success': True, 'message': 'Image generated successfully.', 'image': base64_img_str})
    else:
        return jsonify({'success': False, 'message': 'No matching files found.'})

@app.route('/get-image', methods=['POST'])
def get_image():
    data = request.json
    inputs = data['inputs']

    # Get a list of all files in the directory
    all_files = os.listdir(TRAINED_MODELS_PATH)

    # Filter out files that match the pattern
    matching_files = [file for file in all_files if pattern.match(file)]

    # Load the state_dict from the first matching file
    if matching_files:
        generator_output = generate_image(inputs, matching_files)

        # Save the image to a temporary file
        temp_image_path = f'{BASE_PATH}generated_test_images/temp_generated_image.png'
        save_image(generator_output.data, temp_image_path, nrow=4, normalize=True)
    
        # Convert tensor to PIL Image
        pil_image = torchvision.transforms.ToPILImage()(generator_output[0].detach().cpu())

        # Convert PIL Image to bytes
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format=OUTPUT_IMAGE_FORMAT)
        image_bytes.seek(0)

        # Send the image bytes directly as the response
        # return send_file(image_bytes, mimetype=f'image/{OUTPUT_IMAGE_FORMAT.lower()}')
        return send_file(temp_image_path, mimetype=f'image/{OUTPUT_IMAGE_FORMAT.lower()}')
    else:
        return jsonify({'success': False, 'message': 'No matching files found.'})

if __name__ == '__main__':
    app.run(debug=True)
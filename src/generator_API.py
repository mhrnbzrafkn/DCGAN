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

# INPUT_VECTOR_LENGTH = 128
# NEURALNET_DEEP = 128

TRAINED_MODELS_PATH = f'{BASE_PATH}/'

# Define the pattern to match filenames
PATTERN = re.compile(r'^generator-scripted-\d+\.pt$')

OUTPUT_IMAGE_FORMAT = 'JPEG'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def generate_image(inputs, matching_files):
    matching_file_path = os.path.join(TRAINED_MODELS_PATH, matching_files[0])

    generator = torch.jit.load(matching_file_path)
    generator = generator.to(device)

    # Print model information
    # print(matching_file_path)
    # for param_tensor in generator.state_dict():
    #     print(param_tensor, "\t", generator.state_dict()[param_tensor].size())
        
    generator.eval()

    # Generate image
    generator_inputs = torch.FloatTensor([inputs]).to(device)
    generator_outputs = generator(generator_inputs).to(device)

    return generator_outputs

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/get-image-json', methods=['POST'])
def get_image_json():
    data = request.json
    inputs = data['inputs']
    
    # Get a list of all files in the directory
    all_files = os.listdir(TRAINED_MODELS_PATH)

    # Filter out files that match the pattern
    matching_files = [file for file in all_files if PATTERN.match(file)]

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
    print(all_files)

    # Filter out files that match the pattern
    matching_files = [file for file in all_files if PATTERN.match(file)]

    # Load the state_dict from the first matching file
    if matching_files:
        generator_output = generate_image(inputs, matching_files)

        # Save the image to a temporary file
        temp_image_path = f'{BASE_PATH}/generated_test_images/temp_generated_image.png'
        save_image(generator_output.data, temp_image_path, nrow=4, normalize=True)
    
        # Convert tensor to PIL Image
        pil_image = torchvision.transforms.ToPILImage()(generator_output[0].detach().cuda())

        # Convert PIL Image to bytes
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format=OUTPUT_IMAGE_FORMAT)
        image_bytes.seek(0)

        # Send the image bytes directly as the response
        return send_file(temp_image_path, mimetype=f'image/{OUTPUT_IMAGE_FORMAT.lower()}')
    else:
        return jsonify({'success': False, 'message': 'No matching files found.'})

if __name__ == '__main__':
    app.run(debug=True)
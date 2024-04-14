import io
import re
import os
import uuid
import math
import torch
import torchvision
from torchvision.utils import save_image

base_path = './training-src'
training_model_path = f'{base_path}/output_results'
pattern = re.compile(r'^generator-scripted-\d+\.pt$')
output_image_format = 'JPEG'
input_noise_size = 128
number_of_images_to_generate = 2000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
all_files = os.listdir(training_model_path)
matching_files = [file for file in all_files if pattern.match(file)]
matching_file_path = os.path.join(training_model_path, matching_files[0])
generator = torch.jit.load(matching_file_path)
generator = generator.to(device)
generator.eval()

if matching_files:
    for i in range(number_of_images_to_generate):
        inputs = torch.randn(1, input_noise_size, 1, 1, device=device).to(device)
        generator_output = generator(inputs)
        temp_image_path = f'{base_path}/output_results/generated_test_images/{uuid.uuid4()}.{output_image_format}'
        save_image(generator_output.data, temp_image_path, nrow=4, normalize=True)
        pil_image = torchvision.transforms.ToPILImage()(generator_output[0].detach().cuda())
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format=output_image_format)
        image_bytes.seek(0)
        print(f'Image number {i+1} generated.')
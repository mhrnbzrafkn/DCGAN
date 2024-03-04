import os

# default settings
MODEL_NAME = "model-01"
BASE_PATH = f"./data/output_results/{MODEL_NAME}"
# create base folder
os.makedirs(BASE_PATH, exist_ok=True)

# create a folder to save images that created while training model
GENERATED_IMAGES_PATH = f"{BASE_PATH}/generated_images"
os.makedirs(GENERATED_IMAGES_PATH, exist_ok=True)

# create a folder to save models after training process
TRAINED_MODELS_PATH = f"{BASE_PATH}/trained_models"
os.makedirs(TRAINED_MODELS_PATH, exist_ok=True)

# create a folder to save final report of training process
OUTPUT_REPORT_PATH = f"{BASE_PATH}/output_report"
os.makedirs(OUTPUT_REPORT_PATH, exist_ok=True)

# create a folder to save generated images using trained model
GENERATED_TEST_IMAGES_PATH = f"{BASE_PATH}/generated_test_images"
os.makedirs(GENERATED_TEST_IMAGES_PATH, exist_ok=True)

TRAINING_IMAGES_PATH = "./data/training_images"

# Hyperparameters
INPUT_VECTOR_LENGTH = 10
OUTPUT_IMAGE_SHAPE = (3, 64, 64)
BATCH_SIZE = 512
EPOCHS = 20000
LEARNING_RATE = 0.0001
SAVE_OUTPUT_IMAGE_STEP = 1
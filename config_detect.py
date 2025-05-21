import os
import glob
import torch

# Phase configuration - set to 'detect' for testing/inference
PHASE = 'detect'
VERSION = 'itunet_fed'  

# Model configuration
NUM_CLASSES = 2  # Same as in training
TRANSFORMER_DEPTH = 18  # Same as in the client configuration
INPUT_SHAPE = (384, 384)
CHANNELS = 3

# Device settings
DEVICE = '0'
GPU_NUM = len(DEVICE.split(','))

# Model loading configuration
PRE_TRAINED = True  
CKPT_POINT = False 

# Path to the federated model - use the final global model from training
FED_MODEL_PATH = './fl_communication/global_model/global_model_round_10.pth'  # Adjust round if needed

# Data paths
PATH_DIR = './dataset/detectdata/data_2d'
PATH_LIST = glob.glob(os.path.join(PATH_DIR, '*.hdf5'))
PATH_AP = './dataset/detectdata/data_3d'
AP_LIST = glob.glob(os.path.join(PATH_AP, '*.hdf5'))

# Output paths
OUTPUT_DIR = './results/federated'
LOG_DIR = './log/federated/detect'


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def load_federated_model(model, model_path):
    """Load parameters from a federated global model into a fresh model"""
    if os.path.exists(model_path):
        fed_model_dict = torch.load(model_path, weights_only=False)
        if "parameters" in fed_model_dict:
            params_dict = zip(model.state_dict().keys(), fed_model_dict["parameters"])
            state_dict = {k: torch.Tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict)
            print(f"Successfully loaded federated model from {model_path}")
            return True
        else:
            print(f"Error: No parameters found in the federated model at {model_path}")
    else:
        print(f"Error: Federated model not found at {model_path}")
    return False

# Testing configuration
TEST_CONFIG = {
    'num_classes': NUM_CLASSES,
    'channels': CHANNELS,
    'input_shape': INPUT_SHAPE,
    'batch_size': 1,  # Use batch size 1 for inference
    'num_workers': 4,
    'device': DEVICE,
    'transformer_depth': TRANSFORMER_DEPTH,
    'federated_model_path': FED_MODEL_PATH
}

# Setup for the tester
SETUP_TESTER = {
    'output_dir': OUTPUT_DIR,
    'log_dir': LOG_DIR,
    'phase': PHASE
} 
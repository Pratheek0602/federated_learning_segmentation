# Federated Learning for Medical Image Segmentation

This repository contains a file-based implementation of federated learning for medical image segmentation using the ITUNET model.

## Overview

This implementation uses a file-based communication approach for federated learning, which is suitable for environments where network communication between clients is restricted (e.g., HPC clusters with firewall restrictions).

## Files Structure

- `file_based_server.py`: Server implementation that coordinates training rounds and aggregates model parameters
- `file_based_client.py`: Client implementation that trains models on local data and contributes to the global model
- `fed_detect.py`: Script for running inference/detection using the trained federated model
- `config_detect.py`: Configuration for the detection phase

## File-Based Communication

The implementation uses a shared filesystem structure for communication:
```
./fl_communication/
  ├── global_model/         # Global model storage
  ├── client_updates/       # Client updates by round
  │   └── round_X/         
  └── status/               # Status files for coordination
```

## How to Run

### 1. Training Phase

#### Start the Server:
```bash
python file_based_server.py --num_rounds 30 --min_clients 2 --clean
```

Parameters:
- `--num_rounds`: Number of federated learning rounds
- `--min_clients`: Minimum number of clients required
- `--clean`: Remove previous communication files (optional)

#### Start the Clients:
```bash
# Run in separate terminals or as separate processes
python file_based_client.py --client_id 0 --device "0" --data_path dataset/segdata/data_2d
python file_based_client.py --client_id 1 --device "1" --data_path dataset/segdata/data_2d
```

Parameters:
- `--client_id`: Unique client identifier
- `--device`: GPU device ID
- `--data_path`: Path to the data directory
- `--fold_num`: Number of folds for cross-validation (default: 5)
- `--cur_fold`: Current fold to use (default: 1)

### 2. Detection/Inference Phase

```bash
python fed_detect.py --device "0" --model_path ./fl_communication/global_model/global_model_round_10.pth --data_dir ./dataset/detectdata/data_3d
```

Parameters:
- `--device`: GPU device ID
- `--model_path`: Path to the trained federated model
- `--data_dir`: Directory containing detection data files

See `README_detect.md` for more detailed instructions on the detection phase.

## Data Organization

- Training data: `./dataset/segdata/data_2d/` (2D images)
- Validation data for 3D metrics: `./dataset/segdata/data_3d/`
- Detection data: `./dataset/detectdata/data_3d/`

## Implementation Details

### Data Splitting
- Data is split horizontally by patient ID across clients
- Each client performs 5-fold cross-validation on its local data

### Aggregation Method
- The server uses Federated Averaging (FedAvg) to combine client model updates
- Weighted by the number of samples at each client

### Learning Rate Schedule
- Polynomial decay across rounds
- Additional decay for later rounds to prevent numeric overflow
- Gradient clipping for training stability

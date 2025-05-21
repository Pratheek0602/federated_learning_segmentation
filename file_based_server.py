import argparse
import os
import time
import json
import torch
import numpy as np
from collections import OrderedDict
import shutil
from typing import Dict, List, Tuple

def create_directories():
    """Create necessary directories for file-based communication."""
    os.makedirs("./fl_communication", exist_ok=True)
    os.makedirs("./fl_communication/global_model", exist_ok=True)
    os.makedirs("./fl_communication/client_updates", exist_ok=True)
    os.makedirs("./fl_communication/status", exist_ok=True)

def initialize_global_model(model_path=None):
    """Initialize the global model, either from a checkpoint or randomly."""
    if model_path and os.path.exists(model_path):
        # Load model from checkpoint
        global_model_state = torch.load(model_path, weights_only=False)
        print(f"Loaded global model from {model_path}")
    else:
        # Initialize an empty model (clients will populate parameters structure)
        global_model_state = {"round": 0, "parameters": None}
        print("Initialized empty global model")
        
    # Save initial global model
    torch.save(global_model_state, "./fl_communication/global_model/global_model.pth")
    
    # Create a ready flag file to signal clients
    with open("./fl_communication/status/global_model_ready.txt", "w") as f:
        f.write(f"Round: {global_model_state['round']}")
    
    return global_model_state

def wait_for_client_updates(num_clients, current_round):
    """Wait until all expected client updates are received."""
    print(f"Waiting for {num_clients} client updates for round {current_round}...")
    
    client_updates = []
    # Check for client updates in the client_updates directory
    updates_dir = f"./fl_communication/client_updates/round_{current_round}"
    os.makedirs(updates_dir, exist_ok=True)
    
    timeout = 3600  # 1 hour timeout
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        update_files = [f for f in os.listdir(updates_dir) if f.startswith("client_") and f.endswith(".pth")]
        
        if len(update_files) >= num_clients:
            # All clients have submitted their updates
            for update_file in update_files[:num_clients]:  # Take only the required number of updates
                client_path = os.path.join(updates_dir, update_file)
                try:
                    client_update = torch.load(client_path, weights_only=False)
                    client_updates.append(client_update)
                    print(f"Successfully loaded update from {update_file}")
                except Exception as e:
                    print(f"Error loading client update from {update_file}: {str(e)}")
                    print(f"Waiting for client to resubmit a valid update...")
                    # Remove the corrupted file so the client can retry
                    try:
                        os.remove(client_path)
                        print(f"Removed corrupted file: {client_path}")
                    except:
                        print(f"Could not remove corrupted file: {client_path}")
                    # Continue waiting
                    break
            
            if len(client_updates) >= num_clients:
                print(f"Received updates from {len(client_updates)} clients")
                return client_updates
        
        # Wait before checking again
        time.sleep(5)
    
    # If we get here, we've timed out
    raise TimeoutError(f"Timed out waiting for client updates for round {current_round}")

def aggregate_parameters(client_updates):
    """Aggregate model parameters from all clients (FedAvg)."""
    # Extract parameters and sample counts
    parameters_list = [update["parameters"] for update in client_updates]
    num_samples_list = [update["num_samples"] for update in client_updates]
    
    # Check if all parameter lists have the same structure
    if not all(len(params) == len(parameters_list[0]) for params in parameters_list):
        raise ValueError("All client parameters must have the same structure")
    
    # Calculate total samples
    total_samples = sum(num_samples_list)
    
    # Perform weighted average based on number of samples
    aggregated_parameters = []
    for param_idx in range(len(parameters_list[0])):
        # Initialize with float64 dtype explicitly to avoid casting issues
        weighted_sum = np.zeros_like(parameters_list[0][param_idx], dtype=np.float64)
        
        for client_idx, client_parameters in enumerate(parameters_list):
            weight = num_samples_list[client_idx] / total_samples
            weighted_sum += client_parameters[param_idx] * weight
            
        aggregated_parameters.append(weighted_sum)
    
    return aggregated_parameters

def store_metrics(client_updates, current_round):
    """Store metrics from client updates."""
    metrics = {}
    
    # Aggregate metrics from clients
    for i, update in enumerate(client_updates):
        client_metrics = update.get("metrics", {})
        for key, value in client_metrics.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)
    
    # Calculate average for each metric
    avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
    
    # Store metrics in a JSON file
    metrics_dir = "./fl_communication/status"
    os.makedirs(metrics_dir, exist_ok=True)
    
    with open(os.path.join(metrics_dir, f"round_{current_round}_metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=2)
    
    # Print metrics
    print(f"Round {current_round} Metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.5f}")
    
    return avg_metrics

def run_federated_learning(num_rounds, num_clients):
    """Run the federated learning process."""
    # Setup
    create_directories()
    global_model = initialize_global_model()
    
    # Ensure current_round is an integer
    if isinstance(global_model["round"], list):
        print("Warning: Found round as a list, converting to integer")
        current_round = global_model["round"][0] if global_model["round"] else 0
    else:
        current_round = global_model["round"]
    
    # Create a file to indicate the final round number
    with open("./fl_communication/status/final_round.txt", "w") as f:
        f.write(str(num_rounds))
    
    # Main federated learning loop
    while current_round < num_rounds:
        current_round += 1
        print(f"=== Starting Round {current_round} ===")
        
        # Prepare round directory
        round_dir = f"./fl_communication/client_updates/round_{current_round}"
        os.makedirs(round_dir, exist_ok=True)
        
        # Signal clients to start the round
        with open("./fl_communication/status/current_round.txt", "w") as f:
            f.write(str(current_round))
        
        # Wait for client updates
        client_updates = wait_for_client_updates(num_clients, current_round)
        
        # Aggregate client parameters
        if global_model["parameters"] is None and len(client_updates) > 0:
            # First round, just use the first client's parameters structure
            global_model["parameters"] = client_updates[0]["parameters"]
        else:
            # Aggregate parameters using FedAvg
            global_model["parameters"] = aggregate_parameters(client_updates)
        
        # Store metrics
        metrics = store_metrics(client_updates, current_round)
        
        # Update global model (ensure it's always an integer)
        global_model["round"] = int(current_round)
        
        # Save global model
        model_path = f"./fl_communication/global_model/global_model_round_{current_round}.pth"
        torch.save(global_model, model_path)
        
        # Also save as the latest model
        latest_path = "./fl_communication/global_model/global_model.pth"
        torch.save(global_model, latest_path)
        
        # Signal that a new global model is available
        with open("./fl_communication/status/global_model_ready.txt", "w") as f:
            f.write(f"Round: {current_round}")
        
        print(f"=== Completed Round {current_round}/{num_rounds} ===")
        
    print("Federated learning completed successfully!")
    return global_model

def main():
    parser = argparse.ArgumentParser(description="File-based Federated Learning Server")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of rounds of federated learning")
    parser.add_argument("--min_clients", type=int, default=2, help="Minimum number of clients required")
    parser.add_argument("--clean", action="store_true", help="Clean previous communication files")
    args = parser.parse_args()
    
    if args.clean:
        print("Cleaning previous communication files...")
        if os.path.exists("./fl_communication"):
            shutil.rmtree("./fl_communication")
    
    run_federated_learning(args.num_rounds, args.min_clients)

if __name__ == "__main__":
    main() 
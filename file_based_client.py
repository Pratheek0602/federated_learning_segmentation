import argparse
import os
import glob
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Dict, List, Tuple
from tensorboardX import SummaryWriter

# Import from segmentation module
import sys
sys.path.append('..')
from segmentation.model import itunet_2d
from segmentation.data_loader import (DataGenerator, Normalize, RandomFlip2D,
                                    RandomRotate2D, To_Tensor)
from segmentation.loss import Deep_Supervised_Loss
from segmentation.trainer import AverageMeter, compute_dice
from segmentation.utils import poly_lr

# Custom Compose class for transforms
class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

# Add RunningDice class since it's missing in the original codebase
class RunningDice:
    """
    Running implementation of Dice score.
    """
    def __init__(self, labels, ignore_label=None):
        self.labels = labels
        self.ignore_label = ignore_label

        # Initialize confusion matrix
        self.confusion_matrix = np.zeros((len(self.labels), len(self.labels)))

    def update_matrix(self, ground_truth, prediction):
        """
        Update confusion matrix.
        """
        for i, label in enumerate(self.labels):
            if self.ignore_label is not None and label == self.ignore_label:
                continue
            ref = (ground_truth == i)
            pred = (prediction == i)
            tp = np.logical_and(ref, pred).sum()
            fp = np.logical_and(~ref, pred).sum()
            fn = np.logical_and(ref, ~pred).sum()
            self.confusion_matrix[i, i] += tp
            for j in range(len(self.labels)):
                if i == j:
                    continue
                self.confusion_matrix[i, j] += fp if i == 0 else fn

    def compute_dice(self):
        """
        Compute Dice score based on the confusion matrix.
        """
        dice_list = []
        for i, label in enumerate(self.labels):
            if self.ignore_label is not None and label == self.ignore_label:
                continue
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[i].sum() - tp
            fn = self.confusion_matrix[:, i].sum() - tp
            dice = 2 * tp / (2 * tp + fp + fn + 1e-5)
            dice_list.append(dice)
        
        # Return mean dice and list of dice scores
        return np.mean(dice_list[1:]), dice_list  # Skip background class (index 0)

    def init_op(self):
        """
        Reset the confusion matrix.
        """
        self.confusion_matrix = np.zeros((len(self.labels), len(self.labels)))

class SegmentationClient:
    def __init__(self, train_paths, val_paths, val_ap, cur_fold, client_id, device="0"):
        self.client_id = client_id
        self.device = device
        os.environ['CUDA_VISIBLE_DEVICES'] = self.device
        
        # Setup directories
        self.output_dir = f'./ckpt/client_{client_id}/fold{cur_fold}'
        self.log_dir = f'./log/client_{client_id}/fold{cur_fold}'
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Model parameters
        self.input_shape = (384, 384)
        self.channels = 3
        self.num_classes = 2
        self.transformer_depth = 18
        self.batch_size = 24
        self.num_workers = 4
        self.lr = 1e-3
        self.weight_decay = 0.0001
        self.use_fp16 = False  # Added option for mixed precision training
        
        # Initialize model
        self.net = itunet_2d(n_channels=self.channels, 
                             n_classes=self.num_classes, 
                             image_size=self.input_shape, 
                             transformer_depth=self.transformer_depth)
        self.net = self.net.cuda()
        
        # Initialize loss and optimizer
        self.criterion = Deep_Supervised_Loss().cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), 
                                         lr=self.lr, 
                                         weight_decay=self.weight_decay)
        
        # Setup scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)
        
        # Setup data
        self.train_path = train_paths
        self.val_path = val_paths
        self.val_ap = val_ap
        
        # Setup tensorboard writer
        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0

    def get_parameters(self):
        """Get parameters of the local model."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
    
    def fit(self, server_round, local_epochs=1):
        """Train the model on the local dataset."""
        max_rounds = 30 

        if server_round >= max_rounds:
            base_lr = self.lr * 0.0001
        else:
            base_lr = poly_lr(server_round-1, max_rounds, initial_lr=self.lr)
        
        if server_round > 10:
            base_lr *= 0.1
            
        # Ensure learning rate is always positive and not too small
        base_lr = max(base_lr, 1e-8)
        
        self.optimizer.param_groups[0]['lr'] = base_lr
        print(f"Learning rate for round {server_round}: {base_lr:.8f}")
        
        # Define training transformations
        train_transform = CustomCompose([
            Normalize(),
            RandomRotate2D(),
            RandomFlip2D(mode='hv'),
            To_Tensor(num_class=self.num_classes, input_channel=self.channels)
        ])
        
        # Prepare data loader
        train_dataset = DataGenerator(
            self.train_path, 
            num_class=self.num_classes, 
            transform=train_transform
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
        
        # Training loop
        train_loss_avg = AverageMeter()
        train_dice_avg = AverageMeter()
        run_dice = RunningDice(labels=range(self.num_classes), ignore_label=-1)
        
        for epoch in range(local_epochs):
            self.net.train()
            for step, sample in enumerate(train_loader):
                data = sample['image'].cuda()
                target = sample['label'].cuda()
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    output = self.net(data)
                    if isinstance(output, tuple):
                        output = output[0]
                    loss = self.criterion(output, target)
                
                # Backward and optimize with mixed precision scaling
                self.optimizer.zero_grad()
                if self.use_fp16:
                    self.scaler.scale(loss).backward()
                    # Add gradient clipping to prevent overflow
                    if server_round > 10:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    # Add gradient clipping to prevent overflow
                    if server_round > 10:
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # Calculate metrics
                output = output[0]
                output = output.float()
                loss = loss.float()
                dice = compute_dice(output.detach(), target)
                train_loss_avg.update(loss.item(), data.size(0))
                train_dice_avg.update(dice.item(), data.size(0))
                
                # Update running dice
                output = torch.argmax(torch.softmax(output, dim=1), 1).detach().cpu().numpy()
                target = torch.argmax(target, 1).detach().cpu().numpy()
                run_dice.update_matrix(target, output)
                
                # Log metrics
                self.global_step += 1
                if self.global_step % 10 == 0:
                    rundice, dice_list = run_dice.compute_dice()
                    print(f"Client {self.client_id} - Round {server_round} - Epoch {epoch} - Step {step} - Loss: {loss.item():.5f} - Dice: {dice.item():.5f} - RunDice: {rundice:.5f}")
                    self.writer.add_scalars(
                        f'data/train_loss_dice_client{self.client_id}',
                        {'train_loss': loss.item(), 'train_dice': dice.item()},
                        self.global_step
                    )
        
        # Calculate final metrics for this round
        final_rundice, _ = run_dice.compute_dice()
        
        # Save checkpoint for this round
        if len(self.device.split(',')) > 1:
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
            
        saver = {
            'round': server_round,
            'epoch': local_epochs,
            'save_dir': self.output_dir,
            'state_dict': state_dict,
        }
        
        file_name = f'round_{server_round}_loss_{train_loss_avg.avg:.5f}_dice_{final_rundice:.5f}.pth'
        save_path = os.path.join(self.output_dir, file_name)
        torch.save(saver, save_path)
        print(f"Client {self.client_id} saved checkpoint: {file_name}")
        
        # Return parameters and metrics
        return {
            "parameters": self.get_parameters(),
            "num_samples": len(train_loader.dataset),
            "metrics": {
                "train_loss": train_loss_avg.avg,
                "train_dice": train_dice_avg.avg,
                "train_rundice": float(final_rundice)
            }
        }
    
    def evaluate(self):
        """Evaluate the model on the local validation dataset."""
        # Define validation transformations
        val_transform = CustomCompose([
            Normalize(),
            To_Tensor(num_class=self.num_classes, input_channel=self.channels)
        ])
        
        # Prepare validation data
        val_dataset = DataGenerator(
            self.val_path, 
            num_class=self.num_classes, 
            transform=val_transform
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
        
        # Evaluation metrics
        val_loss_avg = AverageMeter()
        val_dice_avg = AverageMeter()
        run_dice = RunningDice(labels=range(self.num_classes), ignore_label=-1)
        
        # Evaluation loop
        self.net.eval()
        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image'].cuda()
                target = sample['label'].cuda()
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    output = self.net(data)
                    if isinstance(output, tuple):
                        output = output[0]
                    loss = self.criterion(output, target)
                
                # Calculate metrics
                output = output[0]
                output = output.float()
                loss = loss.float()
                
                dice = compute_dice(output.detach(), target)
                val_loss_avg.update(loss.item(), data.size(0))
                val_dice_avg.update(dice.item(), data.size(0))
                
                # Update running dice
                output = torch.argmax(torch.softmax(output, dim=1), 1).detach().cpu().numpy()
                target = torch.argmax(target, 1).detach().cpu().numpy()
                run_dice.update_matrix(target, output)
        
        # Calculate final metrics
        final_rundice, _ = run_dice.compute_dice()
        print(f"Client {self.client_id} - Validation - Loss: {val_loss_avg.avg:.5f} - Dice: {val_dice_avg.avg:.5f} - RunDice: {final_rundice:.5f}")
        
        # Return metrics
        return len(val_loader.dataset), {
            "val_loss": val_loss_avg.avg,
            "val_dice": val_dice_avg.avg,
            "val_rundice": float(final_rundice)
        }

def wait_for_round_start(current_round):
    """Wait until the server signals to start the round."""
    status_file = "./fl_communication/status/current_round.txt"
    
    # Wait for status file to exist
    while not os.path.exists(status_file):
        print(f"Waiting for server to start round {current_round}...")
        time.sleep(5)
    
    # Read round number
    with open(status_file, "r") as f:
        server_round = int(f.read().strip())
        
    return server_round

def check_for_global_model():
    """Check if a global model is available from the server."""
    model_path = "./fl_communication/global_model/global_model.pth"
    if os.path.exists(model_path):
        return model_path
    return None

def save_client_update(client_update, server_round, client_id):
    """Save client update to the shared filesystem."""
    update_dir = f"./fl_communication/client_updates/round_{server_round}"
    os.makedirs(update_dir, exist_ok=True)
    
    # First save to a temporary file
    temp_path = os.path.join(update_dir, f"client_{client_id}_temp.pth")
    update_path = os.path.join(update_dir, f"client_{client_id}.pth")
    
    try:
        # Save to temporary file first
        torch.save(client_update, temp_path)
        
        # If successful, rename to final file name
        if os.path.exists(update_path):
            os.remove(update_path)  # Remove any existing file
        
        os.rename(temp_path, update_path)
        print(f"Saved client {client_id} update for round {server_round}")
    except Exception as e:
        print(f"Error saving client update: {str(e)}")
        # Try to clean up if failed
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False
    
    # Create a flag file to signal that the client has finished
    with open(os.path.join(update_dir, f"client_{client_id}_done.txt"), "w") as f:
        f.write("done")
        
    return True

def run_client(client_id, device, data_path, fold_num=5, cur_fold=1):
    print(f"Starting federated learning client {client_id}...")
    
    # Make sure communication directory exists
    os.makedirs("./fl_communication", exist_ok=True)
    
    # Find all data paths
    all_path_list = glob.glob(os.path.join(data_path, '*.hdf5'))
    all_ap_list = glob.glob(os.path.join('dataset/segdata/data_3d', '*.hdf5'))
    
    if not all_path_list:
        print(f"ERROR: No data found at {data_path}")
        return
    
    # Get sample IDs
    sample_list = list(set([os.path.basename(case).split('_')[0] for case in all_path_list]))
    sample_list.sort()
    
    # Split data based on client ID
    num_clients = 2  # Assuming 2 clients
    total_samples = len(sample_list)
    samples_per_client = total_samples // num_clients
    
    client_samples = []
    for i in range(num_clients):
        if i == num_clients - 1:
            client_samples.append(sample_list[i*samples_per_client:])
        else:
            client_samples.append(sample_list[i*samples_per_client:(i+1)*samples_per_client])
    
    # Cross-validation split for this client
    client_sample_list = client_samples[client_id]
    _len_ = len(client_sample_list) // fold_num
    
    # Split train/validation data
    train_id = []
    validation_id = []
    end_index = cur_fold * _len_
    start_index = end_index - _len_
    
    if cur_fold == fold_num:
        validation_id.extend(client_sample_list[start_index:])
        train_id.extend(client_sample_list[:start_index])
    else:
        validation_id.extend(client_sample_list[start_index:end_index])
        train_id.extend(client_sample_list[:start_index])
        train_id.extend(client_sample_list[end_index:])
    
    # Get train/val paths
    train_path = []
    validation_path = []
    for case in all_path_list:
        if os.path.basename(case).split('_')[0] in train_id:
            train_path.append(case)
        elif os.path.basename(case).split('_')[0] in validation_id:
            validation_path.append(case)
    
    # Get train/val AP paths
    train_ap = []
    validation_ap = []
    for case in all_ap_list:
        if os.path.basename(case).split('_')[0] in train_id:
            train_ap.append(case)
        elif os.path.basename(case).split('_')[0] in validation_id:
            validation_ap.append(case)
    
    print(f"Client {client_id}: {len(train_path)} training samples, {len(validation_path)} validation samples")
    
    # Create client
    client = SegmentationClient(
        train_paths=train_path,
        val_paths=validation_path,
        val_ap=validation_ap,
        cur_fold=cur_fold,
        client_id=client_id,
        device=device
    )
    
    # Main training loop
    current_round = 0
    total_epochs = 160  # Total epochs as per original config
    max_rounds = 10  # Assuming 10 rounds to distribute 160 epochs
    
    while True:
        # Wait for server to signal start of round
        server_round = wait_for_round_start(current_round + 1)
        
        if server_round > current_round:
            print(f"=== Starting Client Round {server_round} ===")
            current_round = server_round
            
            # Check for global model
            global_model_path = check_for_global_model()
            if global_model_path:
                # Load global model parameters
                global_model = torch.load(global_model_path, weights_only=False)
                if global_model["parameters"] is not None:
                    client.set_parameters(global_model["parameters"])
                    print(f"Loaded global model parameters from round {global_model['round']}")
            
            # Train for this round (local epochs determined by server round)
            if server_round >= max_rounds:
                # Use minimum learning rate for rounds beyond max_rounds
                local_epochs = 5
            else:
                local_epochs = total_epochs // max_rounds
                
            # Ensure we don't do too few epochs in any round
            local_epochs = max(local_epochs, 5)
            
            print(f"Training for {local_epochs} local epochs in round {server_round}")
            client_update = client.fit(server_round, local_epochs=local_epochs)
            
            # Save client update to shared filesystem with retry logic
            max_retries = 3
            save_success = False
            
            for attempt in range(max_retries):
                save_success = save_client_update(client_update, server_round, client_id)
                if save_success:
                    break
                else:
                    print(f"Save attempt {attempt+1}/{max_retries} failed. Retrying in 5 seconds...")
                    time.sleep(5)
            
            if not save_success:
                print(f"Failed to save client update after {max_retries} attempts. Will try again in next round.")
            
            # Evaluate model
            num_examples, eval_metrics = client.evaluate()
            print(f"Evaluation metrics for round {server_round}: {eval_metrics}")
            
            # Check if we should exit
            status_file = "./fl_communication/status/global_model_ready.txt"
            if os.path.exists(status_file):
                with open(status_file, "r") as f:
                    content = f.read().strip()
                    if content.startswith("Round:"):
                        latest_round = int(content.split(":")[1].strip())
                        if latest_round >= server_round and latest_round == global_model["round"]:
                            # If this is the final round, break
                            final_round_path = "./fl_communication/status/final_round.txt"
                            if os.path.exists(final_round_path):
                                with open(final_round_path, "r") as f_final:
                                    final_round = int(f_final.read().strip())
                                    if server_round >= final_round:
                                        print(f"Completed final round ({server_round}). Exiting.")
                                        break
        
        # Wait before checking again
        time.sleep(5)
    
    print(f"Client {client_id} completed federated learning.")

def main():
    parser = argparse.ArgumentParser(description="File-based Federated Learning Client")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID")
    parser.add_argument("--data_path", type=str, default="dataset/segdata/data_2d", help="Path to data directory")
    parser.add_argument("--fold_num", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--cur_fold", type=int, default=1, help="Current fold for cross-validation")
    args = parser.parse_args()
    
    run_client(args.client_id, args.device, args.data_path, args.fold_num, args.cur_fold)

if __name__ == "__main__":
    main() 
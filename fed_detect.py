import os
import argparse
import torch
import numpy as np
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from picai_eval import evaluate
import sys

sys.path.append('..')
from segmentation.model import itunet_2d
from segmentation.data_loader import DataGenerator, Normalize, To_Tensor
from config_detect import TEST_CONFIG, load_federated_model, PATH_AP


try:
    from report_guided_annotation import extract_lesion_candidates
except ImportError:
    print("Warning: report_guided_annotation not found. Using fallback extraction method.")
    def extract_lesion_candidates(pred, threshold='dynamic', num_lesions_to_extract=5, min_voxels_detection=10, dynamic_threshold_factor=2.5):
        """Fallback lesion extraction method"""
        if threshold == 'dynamic-fast':
            threshold_value = np.percentile(pred, 98)
        elif threshold == 'dynamic':
            threshold_value = np.percentile(pred, 95)
        else:
            threshold_value = threshold
        binary = pred > threshold_value
        return [binary]

class CustomNormalize2D:
    def __call__(self, sample):
        ct = sample['ct']
        seg = sample['seg']
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                if np.max(ct[i,j]) != 0:
                    ct[i,j] = ct[i,j] / np.max(ct[i,j])
                
        return {'ct': ct, 'seg': seg}

class FederatedTester:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        os.environ['CUDA_VISIBLE_DEVICES'] = self.device
        
        # Initialize model
        self.net = itunet_2d(
            n_channels=config['channels'],
            n_classes=config['num_classes'],
            image_size=config['input_shape'],
            transformer_depth=config['transformer_depth']
        )
        self.net = self.net.cuda()
        
        # Load federated model
        success = load_federated_model(self.net, config['federated_model_path'])
        if not success:
            raise ValueError(f"Could not load federated model from {config['federated_model_path']}")
        
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.use_fp16 = False  # FP16 precision for inference
        
    def detect(self, file_paths, mode='eval'):
        """Run detection on validation data"""
        self.net.eval()
        
        # Setup data transformations
        val_transformer = transforms.Compose([
            CustomNormalize2D(),
            To_Tensor(num_class=self.config['num_classes'], input_channel=self.config['channels'])
        ])
        
        # Setup data loader
        val_dataset = DataGenerator(
            file_paths, 
            num_class=self.config['num_classes'], 
            transform=val_transformer
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Use batch size 1 for inference
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']
                
                data = data.squeeze().transpose(1, 0)
                data = data.cuda()
                target = target.cuda()
                
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    output = self.net(data)
                    if isinstance(output, tuple):
                        output = output[0]
                
                output = output[0]
                output = output.float()
                output = torch.softmax(output, dim=1)
                output = output[:, 0, :, :]  
                output = 1 - output  
                output = output.detach().cpu().numpy()
                
                # Process softmax prediction to detection map
                if mode == 'train':
                    cspca_det_map_npy = extract_lesion_candidates(
                        output, threshold='dynamic-fast')[0]
                else:
                    cspca_det_map_npy = extract_lesion_candidates(
                        output, threshold='dynamic', 
                        num_lesions_to_extract=5,
                        min_voxels_detection=10,
                        dynamic_threshold_factor=2.5)[0]
                
                # Remove secondary concentric/ring detections
                cspca_det_map_npy[cspca_det_map_npy < (np.max(cspca_det_map_npy)/2)] = 0
                
                y_pred.append(cspca_det_map_npy)
                
                target = torch.argmax(target, 1).detach().cpu().numpy().squeeze()
                target[target > 0] = 1
                y_true.append(target)
                
                print(f"Sample {step+1}/{len(val_loader)}: "
                      f"Has lesion: {np.sum(target) > 0}, "
                      f"Max prediction: {np.max(cspca_det_map_npy):.4f}")
                
                torch.cuda.empty_cache()
        
        # Evaluate predictions using picai_eval
        metrics = evaluate(y_pred, y_true)
        print("\nEvaluation Metrics:")
        print(f"AUROC: {metrics.auroc:.4f}")
        print(f"AP: {metrics.AP:.4f}")
        print(f"Score: {metrics.score:.4f}")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Federated Model Detection/Testing")
    parser.add_argument("--device", type=str, default=TEST_CONFIG['device'], 
                        help="GPU device ID")
    parser.add_argument("--model_path", type=str, default=TEST_CONFIG['federated_model_path'], 
                        help="Path to the federated model")
    parser.add_argument("--data_dir", type=str, default=PATH_AP,
                        help="Path to detection data directory")
    args = parser.parse_args()
    
    # Update config with command line arguments
    TEST_CONFIG['device'] = args.device
    TEST_CONFIG['federated_model_path'] = args.model_path
    
    # Create tester
    tester = FederatedTester(TEST_CONFIG)
    
    # Collect the .hdf5 files in the data directory
    data_files = sorted(glob.glob(os.path.join(args.data_dir, "*.hdf5")))
    
    if not data_files:
        print(f"Error: No .hdf5 files found in {args.data_dir}")
        return None
    
    print(f"Found {len(data_files)} .hdf5 files in {args.data_dir}")
    
    # Run detection on validation data
    print("Running detection on validation data...")
    metrics = tester.detect(data_files, mode='eval')
    
    return metrics

if __name__ == "__main__":
    main() 
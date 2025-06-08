import json
import random

from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import Dataset



def build_dataset(data_type, dataset_path, loss_type = 'split'):
    root = Path(dataset_path)
    
    assert root.exists(), f'provided dataset path {root} does not exist'
    
    PATHS = {
        "train": (root / "train"),
        "test": (root / "test"),
        "val": (root / "val"),
    }
    
    data_folder = PATHS[data_type]
    dataset = SceneDataset(data_folder, loss_type)
    return dataset

class SceneDataset(Dataset):
    def __init__(self, npz_folder, loss_type = 'split'):
        """
        Args:
            npz_folder (str): Path to folder containing .npz files
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.npz_folder = npz_folder
        self.files = sorted([f for f in os.listdir(npz_folder) if f.endswith('.npz')])
        self.loss_type = loss_type
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.npz_folder, self.files[idx])
        data = np.load(npz_path, allow_pickle=True)
        
        features_tensors = [torch.from_numpy(f).float() for f in data['features']] 
        
        cleaned_poses = []

        for pose in data['gt_poses']:
            # Convert each 4x4 pose matrix (which currently has dtype=object) to float32 array
            numeric_pose = np.array(pose, dtype=np.float32)
            cleaned_poses.append(numeric_pose)
        
        # Now stack all poses into one numpy array of shape (11, 4, 4)
        poses_np = np.stack(cleaned_poses)
        
        # Convert to torch tensor
        gt_poses_tensor = torch.from_numpy(poses_np)
        
        sample = {
            'boxes': torch.tensor(data['boxes'], dtype=torch.float32),
            'labels': torch.tensor(data['labels'], dtype=torch.int32),
            'confidences': torch.tensor(data['confidences'], dtype=torch.float32),
            'features': features_tensors,
            'strides': torch.tensor(data['strides'], dtype=torch.float32),
            'channels': torch.tensor(data['channels'], dtype=torch.float32),
            'image_path': str(data['image_path'].item()),  # Convert scalar array to string
            'gt_poses': gt_poses_tensor,
            'mask': torch.tensor(data['mask'], dtype=torch.uint8).unsqueeze(0) # 1x640x640  
        }
        pred_objects = torch.cat([sample['boxes'], sample['confidences'].unsqueeze(1), sample['labels'].unsqueeze(1)], dim=1)
        features = sample['features']
        if self.loss_type == 'split':
            target = {'relative_position' : sample['gt_poses'][:,:3,3], 'relative_rotation' : sample['gt_poses'][:,:3,:3]}
        else : 
            target = {'inverse_pose': torch.inverse(sample['gt_poses'])}
        img_mask = sample['mask']

        sample_output = {'features': features, 'prediction': pred_objects, 'img_mask' : img_mask}

        return sample_output, target

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class KLARestorationDataset(Dataset):
    def __init__(self, degraded_dir, gt_dir=None):
        self.degraded_dir = degraded_dir
        self.gt_dir = gt_dir
        
        # Filter to only grab .npy files (ignoring hidden Mac files like .DS_Store)
        self.image_filenames = sorted([f for f in os.listdir(degraded_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.image_filenames)

    def _load_and_format_tensor(self, path):
        # 1. Load the raw numpy array
        arr = np.load(path)
        
        # 2. Convert to PyTorch Tensor
        tensor = torch.from_numpy(arr).float()
        
        # 3. Ensure the tensor has a "Channels" dimension for PyTorch
        # If it's a 2D grayscale array [Height, Width], make it [1, Height, Width]
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        # If it's 3D [Height, Width, Channels], flip it to [Channels, Height, Width]
        elif tensor.dim() == 3 and tensor.shape[-1] in [1, 3]:
            tensor = tensor.permute(2, 0, 1)
            
        return tensor

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        
        # Load degraded tensor
        deg_path = os.path.join(self.degraded_dir, img_name)
        deg_tensor = self._load_and_format_tensor(deg_path)
        
        # Load ground truth tensor (if training)
        if self.gt_dir is not None:
            gt_path = os.path.join(self.gt_dir, img_name)
            gt_tensor = self._load_and_format_tensor(gt_path)
            
            return deg_tensor, gt_tensor, img_name
            
        return deg_tensor, img_name

# --- Quick Test Block ---
if __name__ == "__main__":
    deg_folder = os.path.join("data", "train_degraded")
    gt_folder = os.path.join("data", "train_gt")
    
    dataset = KLARestorationDataset(deg_folder, gt_folder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    deg_batch, gt_batch, filenames = next(iter(dataloader))
    
    print("--- DataLoader Test (NumPy Support) ---")
    print(f"Batch Degraded Shape: {list(deg_batch.shape)} -> (Batch, Channels, Height, Width)")
    print(f"Batch GT Shape      : {list(gt_batch.shape)} -> (Batch, Channels, Height, Width)")
    print(f"Degraded Max Value  : {deg_batch.max().item():.4f} (Notice if it exceeds 1.0!)")
    print(f"Filenames in batch  : {filenames}")
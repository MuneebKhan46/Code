import tensorflow as tf
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
def extract_patches_from_rgb_image(image_path: str, patch_size: int = 224):
    patches = []
    
    if not os.path.exists(image_path):
        print(f"Warning: File {image_path} does not exist.")
        return []

    image = Image.open(image_path)
    if image.mode != 'RGB':
        print(f"Warning: Expected an RGB image, got {image.mode}.")
        return []

    width, height = image.size
    image_array = np.array(image)
  
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image_array[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)), 'constant')
            patches.append(patch)
            
    return patches


class ImageDataset(Dataset):
    def __init__(self, csv_path, original_dir, denoised_dir, patch_size=224):
        df = pd.read_csv(csv_path)
        self.pairs = []
        
        for _, row in df.iterrows():
            original_file_name = f"original_{row['image_name']}.png"
            denoised_file_name = f"denoised_{row['image_name']}.png"
            original_path = os.path.join(original_dir, original_file_name)
            denoised_path = os.path.join(denoised_dir, denoised_file_name)

            original_patches = extract_patches_from_rgb_image(original_path, patch_size)
            denoised_patches = extract_patches_from_rgb_image(denoised_path, patch_size)
            
            self.pairs.extend(zip(original_patches, denoised_patches))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        original_patch, denoised_patch = self.pairs[idx]
        original_patch = torch.from_numpy(original_patch).permute(2, 0, 1).float() / 255.0
        denoised_patch = torch.from_numpy(denoised_patch).permute(2, 0, 1).float() / 255.0
        return original_patch, denoised_patch

original_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/original'
denoised_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/denoised'
csv_path = '/ghosting-artifact-metric/Code/Non_Zeros_Classified_label_filtered.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

dataset = ImageDataset(csv_path, original_dir, denoised_dir, patch_size=224)
train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

psnr_scores, ssim_scores = [], []

with torch.no_grad():
  for inputs, targets in test_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      inputs = inputs.cpu().numpy()
      targets = targets.cpu().numpy()
      
      print(len(inputs))
      for i in range(len(inputs)):
          psnr_scores.append(psnr(targets[i], inputs[i]))
          patch_size = min(inputs[i].shape[0], inputs[i].shape[1])
          win_size = min(7, patch_size)
          if win_size >= 3:
              ssim_val = ssim(targets[i], inputs[i], win_size=win_size, channel_axis=-1, data_range=1.0)
              ssim_scores.append(ssim_val)
          else:
              print(f"Skipping SSIM for patch {i} due to insufficient size")
          
avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores) if ssim_scores else 0

print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}") 


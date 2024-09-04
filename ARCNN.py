# =============================================================================
#  @article{zhang2017beyond,
#    title={Compression Artifacts Reduction by a Deep Convolutional Network},
#    author={Chao Dong, Yubin Deng, Chen Change Loy, and Xiaoou Tang},
#    journal={Proceedings of the  IEEE International Conference on Computer Vision (ICCV)},
#    year={2015}
#  }
# =============================================================================


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


RESULTS_DIR = '/ghosting-artifact-metric/Code/'
num_epochs = 20



class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x


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



# def calculate_metrics(output, target):
#     output_np = output.detach().cpu().numpy().transpose(1, 2, 0)
#     target_np = target.detach().cpu().numpy().transpose(1, 2, 0)

#     psnr_value = psnr(target_np, output_np, data_range=1)
#     ssim_value = ssim(target_np, output_np, multichannel=True)
#     return psnr_value, ssim_value


# def evaluate(model, test_loader, device):
#     model.eval()
#     total_psnr, total_ssim = 0, 0

#     with torch.no_grad():
#         for original, denoised in test_loader:
#             original, denoised = original.to(device), denoised.to(device)
#             output = model(denoised)
#             psnr_value, ssim_value = calculate_metrics(output, original)
#             total_psnr += psnr_value
#             total_ssim += ssim_value
#     return total_psnr / len(test_loader), total_ssim / len(test_loader)


original_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/original'
denoised_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/denoised'
csv_path = '/ghosting-artifact-metric/Code/Non_Zeros_Classified_label_filtered.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

dataset = ImageDataset(csv_path, original_dir, denoised_dir, patch_size=224)
train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


# with strategy.scope():
model = ARCNN()
model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
best_val_loss = float('inf')



for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for original, denoised in train_loader:
        original, denoised = original.to(device), denoised.to(device)
        output = model(denoised)
        loss = criterion(output, original)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss / len(train_loader):.4f}")
  
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
  
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'Best_ARCNN_Model.pth'))
        print(f"New best model saved with validation loss: {val_loss:.4f}")



model.eval()

psnr_scores, ssim_scores = [], []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        for i in range(len(outputs)):
            psnr_scores.append(psnr(targets[i], outputs[i]))
            
            patch_size = min(outputs[i].shape[0], outputs[i].shape[1])
            win_size = min(7, patch_size) 
            
            if win_size >= 3:
                ssim_val = ssim(targets[i], outputs[i], win_size=win_size, channel_axis=-1, data_range=1.0)
                ssim_scores.append(ssim_val)
            else:
                print(f"Skipping SSIM for patch {i} due to insufficient size")

avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores) if ssim_scores else 0

print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")    

# =============================================================================
#  @article{zhang2017beyond,
#    title={Compression Artifacts Reduction by a Deep Convolutional Network},
#    author={Chao Dong, Yubin Deng, Chen Change Loy, and Xiaoou Tang},
#    journal={Proceedings of the  IEEE International Conference on Computer Vision (ICCV)},
#    year={2015}
#  }
# =============================================================================



import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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
        for original_val, denoised_val in val_loader:
            original_val, denoised_val = original_val.to(device), denoised_val.to(device)
            
            outputs_val = model(denoised_val)
            
            loss = criterion(outputs_val, original_val)
            
            val_loss += loss.item()

    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
  
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'Best_ARCNN_Model.pth'))
        print(f"New best model saved with validation loss: {val_loss:.4f}")




model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'Best_ARCNN_Model.pth')))
model.eval()


def visualize_patches(original, denoised, restored):
    if isinstance(original, np.ndarray):
        original = torch.tensor(original)
    if isinstance(denoised, np.ndarray):
        denoised = torch.tensor(denoised)
    if isinstance(restored, np.ndarray):
        restored = torch.tensor(restored)
    

    original = (original * 255).clamp(0, 255).byte()
    denoised = (denoised * 255).clamp(0, 255).byte()
    restored = (restored * 255).clamp(0, 255).byte()
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Original Image")
    axs[1].imshow(denoised.permute(1, 2, 0).cpu().numpy())
    axs[1].set_title("Denoised Image")
    axs[2].imshow(restored.permute(1, 2, 0).cpu().numpy())
    axs[2].set_title("ARCNN")
    for ax in axs:
        ax.axis('off')
    plt.show()

psnr_scores, ssim_scores = [], []
visualized_images = 0
visualize_limit = 10


with torch.no_grad():
    for original_test, denoised_test in test_loader:
        original_test, denoised_test = original_test.to(device), denoised_test.to(device)
        
        outputs_test = model(denoised_test)
        
        outputs_test = outputs_test.cpu().numpy()
        original_test = original_test.cpu().numpy()

        for i in range(len(outputs_test)):
            
            psnr_scores.append(psnr(original_test[i], outputs_test[i]))
            
            patch_size = min(outputs_test[i].shape[0], outputs_test[i].shape[1])
            win_size = min(7, patch_size) 
            
            if win_size >= 3:
                ssim_val = ssim(original_test[i], outputs_test[i], win_size=win_size, channel_axis=-1, data_range=1.0)
                ssim_scores.append(ssim_val)
            else:
                print(f"Skipping SSIM for patch {i} due to insufficient size")

            if visualized_images < visualize_limit:
                visualize_patches(original_test[i], denoised_test[i], outputs_test[i])
                visualized_images += 1

   


avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores) if ssim_scores else 0

print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}") 

# Average PSNR: 33.0046
# Average SSIM: 0.9213
















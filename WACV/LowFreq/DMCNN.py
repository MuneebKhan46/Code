import cv2
import os
import torch
import shutil
import logging
from glob import glob
import numpy as np
import torch.nn as nn
from PIL import Image
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
PATCH_SIZE = 224
BATCH_SIZE = 20
LEARNING_RATE = 1e-4
weight_decay = 1e-4
EPOCHS = 10
COLOR_CHANNELS = 3
RESULTS_DIR = '/ghosting-artifact-metric/Code/'
CHECKPOINT_INTERVAL = 5

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


class CustomDataset(Dataset):
    def __init__(self, original_dir, denoised_dir, csv_path, transform=None):
        self.original_dir = original_dir
        self.denoised_dir = denoised_dir
        self.transform = transform

        self.all_original_patches, self.all_denoised_patches = load_data_from_csv(csv_path, original_dir, denoised_dir)

    def __len__(self):
        return len(self.all_original_patches)

    def __getitem__(self, idx):
        original_patch = self.all_original_patches[idx]
        denoised_patch = self.all_denoised_patches[idx]

        original_patch = Image.fromarray(original_patch)
        denoised_patch = Image.fromarray(denoised_patch)

        if self.transform:
            original_patch = self.transform(original_patch)
            denoised_patch = self.transform(denoised_patch)

        return original_patch, denoised_patch


def extract_patches_from_rgb_image(image_path: str, patch_size: int = 224):
    patches = []

    if not os.path.exists(image_path):
        print(f"Warning: File {image_path} does not exist.")
        return [], []

    image = Image.open(image_path)
    if image.mode != 'RGB':
        print(f"Warning: Expected an RGB image, got {image.mode}.")
        return [], []

    width, height = image.size
    image_array = np.array(image)

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image_array[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]),
                               (0, patch_size - patch.shape[1]), (0, 0)), 'constant')
            patches.append(patch)

    return patches


def load_data_from_csv(csv_path, original_dir, denoised_dir):
    df = pd.read_csv(csv_path)

    all_original_patches = []
    all_denoised_patches = []

    for _, row in df.iterrows():
        original_file_name = f"original_{row['image_name']}.png"
        denoised_file_name = f"denoised_{row['image_name']}.png"

        original_path = os.path.join(original_dir, original_file_name)
        denoised_path = os.path.join(denoised_dir, denoised_file_name)

        original_patches = extract_patches_from_rgb_image(original_path)
        denoised_patches = extract_patches_from_rgb_image(denoised_path)

        all_original_patches.extend(original_patches)
        all_denoised_patches.extend(denoised_patches)

    return all_original_patches, all_denoised_patches


transform = transforms.Compose([
    transforms.ToTensor(),
])

original_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/original'
denoised_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/denoised'
csv_path = '/ghosting-artifact-metric/Code/Non_Zeros_Classified_label_filtered.csv'


dataset = CustomDataset(original_dir, denoised_dir, csv_path, transform=transform)

train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class DCTLayer(nn.Module):
    def __init__(self):
        super(DCTLayer, self).__init__()
        self.register_buffer('dct_matrix', self.create_dct_matrix(IMAGE_SIZE))

    def create_dct_matrix(self, N):
        dct_matrix = np.zeros((N, N))
        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct_matrix[k, n] = np.sqrt(1/N)
                else:
                    dct_matrix[k, n] = np.sqrt(2/N) * np.cos(np.pi * k * (2*n+1) / (2*N))
        return torch.FloatTensor(dct_matrix)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert height == IMAGE_SIZE and width == IMAGE_SIZE, "Input dimensions must match the defined IMAGE_SIZE"

        x_reshaped = x.view(batch_size * channels, height, width)
        dct_output = torch.matmul(self.dct_matrix, torch.matmul(x_reshaped, self.dct_matrix.t()))
        dct_output = dct_output.view(batch_size, channels, height, width)
        return dct_output


class IDCTLayer(nn.Module):
    def __init__(self):
        super(IDCTLayer, self).__init__()
        self.register_buffer('idct_matrix', self.create_dct_matrix(IMAGE_SIZE).t())

    def create_dct_matrix(self, N):
        dct_matrix = np.zeros((N, N))
        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct_matrix[k, n] = np.sqrt(1/N)
                else:
                    dct_matrix[k, n] = np.sqrt(2/N) * np.cos(np.pi * k * (2*n+1) / (2*N))
        return torch.FloatTensor(dct_matrix)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x_reshaped = x.view(batch_size * channels, height, width)
        idct_output = torch.matmul(self.idct_matrix, torch.matmul(x_reshaped, self.idct_matrix.t()))
        idct_output = idct_output.view(batch_size, channels, height, width)
        return idct_output


class DMCNN(nn.Module):
    def __init__(self):
        super(DMCNN, self).__init__()
        self.dct_layers = nn.Sequential(
            DCTLayer(),
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            # Final conv layer to reduce channels from 64 to 3
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

        self.pixel_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

        
        self.residual_weight = nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, x):
        dct_output = self.dct_layers(x)
        pixel_output = self.pixel_layers(x)
        output = self.residual_weight * dct_output + (1 - self.residual_weight) * pixel_output
        return output



model = DMCNN()
model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
early_stopping_patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
  model.train()
  train_loss = 0.0
  for original, denoised in train_loader:
    original, denoised = original.to(device), denoised.to(device
    optimizer.zero_grad()
    outputs = model(denoised)
    loss = criterion(outputs, original)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()

  train_loss /= len(train_loader)
  print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss:.4f}")

  model.eval()
  val_loss = 0.0
  with torch.no_grad():
    for original_val, denoised_val in val_loader:
      original_val, denoised_val = original_val.to(device), denoised_val.to(device)
      outputs_val = model(denoised_val)
      loss = criterion(outputs_val, original_val)
      val_loss += loss.item()
      
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'Best_DMCNN_model2.pth'))
        print(f"New best model saved with validation loss: {val_loss:.4f}")
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

    scheduler.step(val_loss)

model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'Best_DMCNN_model2')))
model.eval()

psnr_scores, ssim_scores = [], []
results = []

with torch.no_grad():
  for original_test, denoised_test in test_loader:
    original_test, denoised_test = original_test.to(device), denoised_test.to(device)
    outputs_test = model(denoised_test)
    outputs_test = outputs_test.cpu().numpy()
    original_test = original_test.cpu().numpy()
    for i in range(len(outputs_test)):
      psnr_scores.append(psnr(original_test[i], outputs_test[I]))
      patch_size = min(outputs_test[i].shape[0], outputs_test[i].shape[1])
      win_size = min(7, patch_size)
      if win_size >= 3:
        ssim_val = ssim(original_test[i], outputs_test[i], win_size=win_size, channel_axis=-1, data_range=1.0)
        ssim_scores.append(ssim_val)
      else:
        print(f"Skipping SSIM for patch {i} due to insufficient size")

avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores) if ssim_scores else 0

print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")

results_csv_path = os.path.join(Results_dir, 'results.csv')

results.append({'Model': 'DMCNN', 'Artifact Type': 'Low Freq', 'PSNR': avg_psnr, 'SSIM': avg_ssim})

if os.path.exists(results_csv_path):
  df_existing = pd.read_csv(results_csv_path)
  df_new = pd.DataFrame(results)
  df_results = pd.concat([df_existing, df_new], ignore_index=True)
else:
  df_results = pd.DataFrame(results)
  
df_results.to_csv(results_csv_path, index=False)

print(f"Results saved to {results_csv_path}")

# =============================================================================
#  @article{zhang2017beyond,
#    title={BlockCNN: A Deep Network for Artifact Removal and Image Compression},
#    author={Danial Maleki, Soheila Nadalian, Mohammad Mahdi Derakhshani, Mohammad Amin Sadeghi},
#    journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
#    year={2018}, 
#    pages={2555-2558}, 
#  }
# =============================================================================



import os
import cv2
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

device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

IMAGE_SIZE = 224
PATCH_SIZE = 224
BATCH_SIZE = 24
EPOCHS = 50
LEARNING_RATE = 1e-3
weight_decay = 1e-4

COLOR_CHANNELS = 3
CHECKPOINT_INTERVAL = 5 

Results_dir = '/ghosting-artifact-metric/WACV/Result'
if not os.path.exists(Results_dir):
    os.makedirs(Results_dir)

model_dir = '/ghosting-artifact-metric/WACV/Model'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

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
        return []

    image = Image.open(image_path)
    if image.mode != 'RGB':
        print(f"Warning: Expected an RGB image, got {image.mode}.")
        return []

    width, height = image.size
    image_array = np.array(image)
    patch_number = 0

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image_array[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)), 'constant')
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
csv_path = '/ghosting-artifact-metric/Code/WACV/HighFreq/high_frequency_classification_label.csv'


dataset = CustomDataset(original_dir, denoised_dir, csv_path, transform=transform)

train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class BottleNeck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = F.relu(out)

        return out

class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        k = 64
        self.conv_1 = nn.Conv2d(COLOR_CHANNELS, k, (3, 5), (1, 1), padding=(1, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(k)

        self.layer_1 = BottleNeck(k, k)
        self.layer_2 = BottleNeck(k, k)


        self.conv_2 = nn.Conv2d(k, k*2, (3, 5), (1, 1), padding=(1, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(k*2)

        self.layer_3 = BottleNeck(k*2, k*2)

        self.conv_3 = nn.Conv2d(k*2, k*4, (1, 5), (1, 1), padding=(0, 2), bias=False)
        self.bn3 = nn.BatchNorm2d(k*4)

        self.layer_4 = BottleNeck(k*4, k*4)
        self.layer_5 = BottleNeck(k*4, k*4)


        self.conv_4 = nn.Conv2d(k*4, k*8, (1, 1), (1, 1), padding=(0, 0), bias=False)
        self.bn4 = nn.BatchNorm2d(k*8)
        
        self.layer_6 = BottleNeck(k*8, k*8)

        self.conv_5 = nn.Conv2d(k*8, k*4, 1, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(k*4)

        self.layer_7 = BottleNeck(k*4, k*4)

        self.conv_6 = nn.Conv2d(k*4, k*2, 1, 1, 0, bias=False)
        self.bn6 = nn.BatchNorm2d(k*2)

        self.layer_8 = BottleNeck(k*2, k*2)
        
        self.conv_7 = nn.Conv2d(k*2, k, 1, 1, 0, bias=False)
        self.bn7 = nn.BatchNorm2d(k)

        self.layer_9 = BottleNeck(k, k)
        
        # self.conv_8 = nn.Conv2d(k*2, COLOR_CHANNELS, 1, 1, 0, bias=False)
        
        self.conv_8 = nn.Conv2d(k, COLOR_CHANNELS, 1, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = x.squeeze(1)  
        out = F.relu(self.bn1(self.conv_1(x)))
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = F.relu(self.bn2(self.conv_2(out)))
        out = self.layer_3(out)
        out = F.relu(self.bn3(self.conv_3(out)))
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = F.relu(self.bn4(self.conv_4(out)))
        out = self.layer_6(out)
        out = F.relu(self.bn5(self.conv_5(out)))
        out = self.layer_7(out)
        out = F.relu(self.bn6(self.conv_6(out)))
        out = self.layer_8(out)  
        out = F.relu(self.bn7(self.conv_7(out)))
        out = self.layer_9(out)  
        out = self.conv_8(out)   
        out = self.sig(out)      
        out = out * 255

        # out = torch.sigmoid(self.conv_8(out))  
    
        return out


model = CNN_Net()
model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

early_stopping_patience = 10
best_val_loss = float('inf')
early_stopping_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for original, denoised in train_loader:
        original, denoised = original.to(device), denoised.to(device)
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
        early_stopping_counter = 0  # Reset counter when improvement happens
        torch.save(model.state_dict(), os.path.join(model_dir, 'HighFreq_BlockCNN_Model.pth'))
        print(f"New best model saved with validation loss: {val_loss:.4f}")
    else:
        early_stopping_counter += 1 
        print(f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

    scheduler.step(val_loss)

    for param_group in optimizer.param_groups:
        print(f"Current learning rate: {param_group['lr']}")

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
# early_stopping_patience = 10
# best_val_loss = float('inf')
# epochs_no_improve = 0

# for epoch in range(EPOCHS):
#     model.train()
#     train_loss = 0.0
#     for original, denoised in train_loader:
#         original, denoised = original.to(device), denoised.to(device)
#         optimizer.zero_grad()
        
#         outputs = model(denoised)
        
#         loss = criterion(outputs, original)
        
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     train_loss /= len(train_loader)
#     print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss:.4f}")

#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for original_val, denoised_val in val_loader:
#             original_val, denoised_val = original_val.to(device), denoised_val.to(device)
            
#             outputs_val = model(denoised_val)
            
#             loss = criterion(outputs_val, original_val)
#             val_loss += loss.item()

#     val_loss /= len(val_loader)
#     print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss:.4f}")

#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         early_stopping_counter = 0
#         torch.save(model.state_dict(), os.path.join(model_dir, 'HighFreq_BlockCNN_Model.pth'))
#         print(f"New best model saved with validation loss: {val_loss:.4f}")
#     else:
#         early_stopping_counter += 1

#     if early_stopping_counter >= early_stopping_patience:
#         print("Early stopping triggered.")
#         break

#     scheduler.step(val_loss)


model.load_state_dict(torch.load(os.path.join(model_dir, 'HighFreq_BlockCNN_Model.pth')))
model.eval()

def save_image(image_tensor, filename):
    image_array = (image_tensor * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    image_pil = Image.fromarray(image_array)
    image_pil.save(filename)


def visualize_and_save_patches(original, denoised, restored, idx):
    if isinstance(original, np.ndarray):
        original = torch.tensor(original)
    if isinstance(denoised, np.ndarray):
        denoised = torch.tensor(denoised)
    if isinstance(restored, np.ndarray):
        restored = torch.tensor(restored)
    
    original_file = os.path.join(image_save_dir, f"BlockCNN_original_patch_{idx}.png")
    denoised_file = os.path.join(image_save_dir, f"BlockCNN_denoised_patch_{idx}.png")
    restored_file = os.path.join(image_save_dir, f"BlockCNN_restored_patch_{idx}.png")
    
    save_image(original, original_file)
    save_image(denoised, denoised_file)
    save_image(restored, restored_file)

    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # axs[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    # axs[0].set_title("Original Image")
    # axs[1].imshow(denoised.permute(1, 2, 0).cpu().numpy())
    # axs[1].set_title("Denoised Image")
    # axs[2].imshow(restored.permute(1, 2, 0).cpu().numpy())
    # axs[2].set_title("ARCNN")
    # for ax in axs:
    #     ax.axis('off')
    # plt.show()


psnr_scores, ssim_scores = [], []
results = []
image_save_dir = os.path.join(Results_dir, 'images/BlockCNN_High/')
os.makedirs(image_save_dir, exist_ok=True)

visualized_images = 0
visualize_limit = 2

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
                visualize_and_save_patches(original_test[i], denoised_test[i], outputs_test[i], visualized_images)
                visualized_images += 1




avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores) if ssim_scores else 0

print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")


results_csv_path = os.path.join(Results_dir, 'results.csv')

results.append({'Model': 'Block-CNN', 'Artifact Type': 'High Freq', 'PSNR': avg_psnr, 'SSIM': avg_ssim})

if os.path.exists(results_csv_path):
    df_existing = pd.read_csv(results_csv_path)
    df_new = pd.DataFrame(results)
    df_results = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_results = pd.DataFrame(results)
  
df_results.to_csv(results_csv_path, index=False)

print(f"Results saved to {results_csv_path}")

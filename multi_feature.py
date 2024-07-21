import os
from os import path
import csv
import cv2
import textwrap
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

from keras.models import Sequential

from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate

from keras.callbacks import ModelCheckpoint,EarlyStopping

from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

models = []
class_1_accuracies = []

original_dir = '/Dataset/dataset_patch_raw_ver3/original'
denoised_dir = '/Dataset/dataset_patch_raw_ver3/denoised'
csv_path     = '/Dataset/patch_label_median_verified3.csv'
result_file_path = "/Code/Results/Multi_Feature_result.csv"

def extract_y_channel_from_yuv_with_patch_numbers(yuv_file_path: str, width: int, height: int):
    y_size = width * height
    patches, patch_numbers = [], []

    if not os.path.exists(yuv_file_path):
        print(f"Warning: File {yuv_file_path} does not exist.")
        return [], []

    with open(yuv_file_path, 'rb') as f:
        y_data = f.read(y_size)

    if len(y_data) != y_size:
        print(f"Warning: Expected {y_size} bytes, got {len(y_data)} bytes.")
        return [], []

    y_channel = np.frombuffer(y_data, dtype=np.uint8).reshape((height, width))
    patch_number = 0

    for i in range(0, height, 224):
        for j in range(0, width, 224):
            patch = y_channel[i:i+224, j:j+224]
            if patch.shape[0] < 224 or patch.shape[1] < 224:
                patch = np.pad(patch, ((0, 224 - patch.shape[0]), (0, 224 - patch.shape[1])), 'constant')
            patches.append(patch)
            patch_numbers.append(patch_number)
            patch_number += 1

    return patches, patch_numbers
  

def load_data_from_csv(csv_path, original_dir, denoised_dir):
    df = pd.read_csv(csv_path)
    
    all_original_patches = []
    all_denoised_patches = []
    all_scores = []
    denoised_image_names = []
    all_patch_numbers = []

    for _, row in df.iterrows():
        
        original_file_name = f"original_{row['image_name']}.raw"
        denoised_file_name = f"denoised_{row['image_name']}.raw"

        original_path = os.path.join(original_dir, original_file_name)
        denoised_path = os.path.join(denoised_dir, denoised_file_name)
        
        original_patches, original_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(original_path, row['width'], row['height'])
        denoised_patches, denoised_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(denoised_path, row['width'], row['height'])

        all_original_patches.extend(original_patches)
        all_denoised_patches.extend(denoised_patches)
        denoised_image_names.extend([row['image_name']] * len(denoised_patches))
        all_patch_numbers.extend(denoised_patch_numbers) 


        scores = np.array([0 if float(score) == 0 else 1 for score in row['patch_score'].split(',')])
        if len(scores) != len(original_patches) or len(scores) != len(denoised_patches):
            print(f"Error: Mismatch in number of patches and scores for {row['image_name']}")
            continue
        all_scores.extend(scores)

    return all_original_patches, all_denoised_patches, all_scores, denoised_image_names, all_patch_numbers



def calculate_difference(original, denoised):
    return [denoised_patch.astype(np.int16) - orig_patch.astype(np.int16) for orig_patch, denoised_patch in zip(original, denoised)]

def calculate_normalized_difference(original, denoised):
    epsilon = 1e-6
    return [(denoised_patch.astype(np.float32) - orig_patch.astype(np.float32)) / (orig_patch.astype(np.float32) + denoised_patch.astype(np.float32) + epsilon) for orig_patch, denoised_patch in zip(original, denoised)]

def calculate_psnr(original, denoised):
    return [peak_signal_noise_ratio(orig_patch, denoised_patch, data_range=255) for orig_patch, denoised_patch in zip(original, denoised)]

def calculate_ssim(original, denoised):
    return [structural_similarity(orig_patch, denoised_patch, data_range=255) for orig_patch, denoised_patch in zip(original, denoised)]


# def combine_features(diff_patches, normalized_diff_patches):
#     combined_features = [np.stack((diff, norm_diff), axis=-1) for diff, norm_diff in zip(diff_patches, normalized_diff_patches)]
#     return combined_features


def combine_features(diff_patches, normalized_diff_patches, psnr_values, ssim_values):
    combined_features = []
    for i in range(len(diff_patches)):
        feature_map = np.stack((diff_patches[i], normalized_diff_patches[i]), axis=-1)
        psnr_map = np.full((224, 224, 1), psnr_values[i])
        ssim_map = np.full((224, 224, 1), ssim_values[i])
        combined_feature = np.concatenate((feature_map, psnr_map, ssim_map), axis=-1)
        combined_features.append(combined_feature)
    return combined_features


def prepare_data(data, labels):
    data = np.array(data).astype('float32')
    lbl = np.array(labels)
    return data, lbl

def save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path):
    if os.path.exists(result_file_path):
        df_existing = pd.read_csv(result_file_path)
        df_row = pd.DataFrame({
            'Model': [model_name],
            'Technique': [technique],
            'Feature Map': [feature_name],
            'Overall Accuracy': [test_acc],
            'Precision': [weighted_precision],
            'Recall': [weighted_recall],
            'F1-Score': [weighted_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })
        df_metrics = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_metrics = pd.DataFrame({
            'Model': [model_name],
            'Technique': [technique],
            'Feature Map': [feature_name],
            'Overall Accuracy': [test_acc],
            'Precision': [weighted_precision],
            'Recall': [weighted_recall],
            'F1-Score': [weighted_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })

    df_metrics.to_csv(result_file_path, index=False)

def create_cnn_model(input_shape=(224, 224, 4)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='elu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())     
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='elu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dense(2, activation='softmax'))
    return model


original_patches, denoised_patches, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(csv_path, original_dir, denoised_dir)


diff_patches = calculate_difference(original_patches, denoised_patches)

normalized_diff_patches = calculate_normalized_difference(original_patches, denoised_patches)

psnr_values = calculate_psnr(original_patches, denoised_patches)

ssim_values =calculate_ssim( original_patches, denoised_patches)

combine_features = combine_features(diff_patches, normalized_diff_patches, psnr_values, ssim_values)

print(f" Difference Map Shape: {diff_patches[0].shape}")
print(f" Normalized Difference Map Shape: {normalized_diff_patches[0].shape}")
print(f" PSNR Map Shape: {psnr_values[0].shape}")
print(f" SSIM Map Shape: {ssim_values[0].shape}")
print(f" Combine Feature Map Shape: {combined_features[0].shape}")


combined_features_np, labels_np = prepare_data(combined_features, labels)

print(f" Combine Feature Shape: {combined_features_np.shape}")


combined = list(zip(combined_features_np, labels_np))
combined = sklearn_shuffle(combined)


ghosting_artifacts = [item for item in combined if item[1] == 1]
non_ghosting_artifacts = [item for item in combined if item[1] == 0]

num_ghosting_artifacts = len(ghosting_artifacts)
num_non_ghosting_artifacts = len(non_ghosting_artifacts)

print(f" Total GA Patches: {num_ghosting_artifacts}")
print(f" Total NGA Labels: {num_non_ghosting_artifacts}")



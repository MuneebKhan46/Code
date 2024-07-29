import os
from os import path
import csv

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

original_dir = '/ghosting-artifact-metric/dataset/dataset_patch_raw_ver3/original'
denoised_dir = '/ghosting-artifact-metric/dataset/dataset_patch_raw_ver3/denoised'
csv_path     = '/ghosting-artifact-metric/dataset/patch_label_median_verified3.csv'

result_file_path = "/ghosting-artifact-metric/Project/Results/Result.csv"

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

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

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

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

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

def calculate_gradient_map(patches):
    gradient_maps = []
    for patch in patches:
        sobel_x = sobel_h(patch.astype(np.float32))
        sobel_y = sobel_v(patch.astype(np.float32))
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_maps.append(gradient_magnitude)
    return gradient_maps


#########################################################################################################################################################################################################################################

# def calculate_gradient_difference_map(original_gradient_maps, denoised_gradient_maps):
#     difference_maps = []
#     for orig_grad, denoised_grad in zip(original_gradient_maps, denoised_gradient_maps):
#         difference = (orig_grad - denoised_grad)        
#         difference_maps.append(difference)
#     return difference_maps

#########################################################################################################################################################################################################################################

def calculate_difference(original, denoised):
    return [denoised_patch.astype(np.int16) - orig_patch.astype(np.int16) for orig_patch, denoised_patch in zip(original, denoised)]


#########################################################################################################################################################################################################################################

def combine_features(denoised_patches, diff_patches, denoised_gradient_maps):
    combined_features = [np.stack((denoised, diff, grad), axis=-1) for denoised, diff, grad in zip(denoised_patches, diff_patches, denoised_gradient_maps)]
    return combined_features
    
#########################################################################################################################################################################################################################################

def prepare_data(data, labels):
    data = np.array(data).astype('float32') / 255.0
    lbl = np.array(labels)
    return data, lbl

#########################################################################################################################################################################################################################################

def save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, macro_precision, macro_recall, macro_f1_score, micro_precision, micro_recall, micro_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path):
    function = "Sigmoid"
    if path.exists(result_file_path):
    
        df_existing = pd.read_csv(result_file_path)
        df_new_row = pd.DataFrame({
            'Model': [model_name],
            'Technique' : [technique],
            'Feature Map' : [feature_name],
            'Function' : [function],
            'Overall Accuracy': [test_acc],
            'Weight Precision': [weighted_precision],
            'Weight Recall': [weighted_recall],
            'Weight F1-Score': [weighted_f1_score],
            'Macro Precision': [macro_precision],
            'Macro Recall': [macro_recall],
            'Macro F1-Score': [macro_f1_score],
            'Micro Precision': [micro_precision],
            'Micro Recall': [micro_recall],
            'Micro F1-Score': [micro_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })
        df_metrics = pd.concat([df_existing, df_new_row], ignore_index=True)
    else:
 
        df_metrics = pd.DataFrame({
            'Model': [model_name],
            'Technique' : [technique],
            'Feature Map' : [feature_name],
            'Function' : [function],
            'Overall Accuracy': [test_acc],
            'Weight Precision': [weighted_precision],
            'Weight Recall': [weighted_recall],
            'Weight F1-Score': [weighted_f1_score],
            'Macro Precision': [macro_precision],
            'Macro Recall': [macro_recall],
            'Macro F1-Score': [macro_f1_score],
            'Micro Precision': [micro_precision],
            'Micro Recall': [micro_recall],
            'Micro F1-Score': [micro_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })

    df_metrics.to_csv(result_file_path, index=False)

#########################################################################################################################################################################################################################################

def augmented_images(data, num_augmented_images_per_original):
    augmented_images = []
    
    data_augmentation = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    for i, patch in enumerate(data):
        patch = np.expand_dims(patch, axis=0)
        temp_generator = data_augmentation.flow(patch, batch_size=1)
        
        for _ in range(num_augmented_images_per_original):
            augmented_image = next(temp_generator)[0]  
            augmented_image = np.squeeze(augmented_image)
            augmented_images.append(augmented_image)
    return augmented_images

#########################################################################################################################################################################################################################################

def create_cnn_model(input_shape=(224, 224, 3)):
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
    model.add(Dense(1, activation='sigmoid'))
    return model


#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

original_patches, denoised_patches, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(csv_path, original_dir, denoised_dir)

diff_patches = calculate_difference(original_patches, denoised_patches)
denoised_gradient_maps = calculate_gradient_map(denoised_patches)

combined_feature = combine_features(denoised_patches, diff_patches, denoised_gradient_maps)

combined_feature_np, labels_np = prepare_data(combined_feature, labels)

print(f" Combine Feature Shape: {combined_features_np.shape}")


#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

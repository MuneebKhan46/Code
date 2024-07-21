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

original_dir = '/Dataset/dataset_patch_raw_ver3/original'
denoised_dir = '/Dataset/dataset_patch_raw_ver3/denoised'
csv_path     = '/Dataset/patch_label_median_verified3.csv'
result_file_path = "/Code/Results/Multi_Feature_result.csv"

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

#########################################################################################################################################################################################################################################

def combine_features(diff_patches, normalized_diff_patches, psnr_values, ssim_values):
    combined_features = []
    for i in range(len(diff_patches)):
        feature_map = np.stack((diff_patches[i], normalized_diff_patches[i]), axis=-1)
        psnr_map = np.full((224, 224, 1), psnr_values[i])
        ssim_map = np.full((224, 224, 1), ssim_values[i])
        combined_feature = np.concatenate((feature_map, psnr_map, ssim_map), axis=-1)
        combined_features.append(combined_feature)
    return combined_features

#########################################################################################################################################################################################################################################

def prepare_data(data, labels):
    data = np.array(data).astype('float32')
    lbl = np.array(labels)
    return data, lbl

#########################################################################################################################################################################################################################################

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
    model.add(Dense(1, activation='sigmoid'))
    return model


#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

original_patches, denoised_patches, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(csv_path, original_dir, denoised_dir)

diff_patches = calculate_difference(original_patches, denoised_patches)
normalized_diff_patches = calculate_normalized_difference(original_patches, denoised_patches)
psnr_values = calculate_psnr(original_patches, denoised_patches)
ssim_values =calculate_ssim( original_patches, denoised_patches)

combine_features = combine_features(diff_patches, normalized_diff_patches, psnr_values, ssim_values)

print(f" Difference Map Shape: {diff_patches[0].shape}")
print(f" Normalized Difference Map Shape: {normalized_diff_patches[0].shape}")
print(f" Combine Feature Map Shape: {combine_features[0].shape}")


combined_features_np, labels_np = prepare_data(combine_features, labels)

print(f" Combine Feature Shape: {combined_features_np.shape}")


#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

combined = list(zip(combined_features_np, labels_np))
combined = sklearn_shuffle(combined)


ghosting_artifacts = [item for item in combined if item[1] == 1]
non_ghosting_artifacts = [item for item in combined if item[1] == 0]

num_ghosting_artifacts = len(ghosting_artifacts)
num_non_ghosting_artifacts = len(non_ghosting_artifacts)

print(f" Total GA Patches: {num_ghosting_artifacts}")
print(f" Total NGA Labels: {num_non_ghosting_artifacts}")

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

num_test_ghosting = 1500
num_test_non_ghosting = 1500

num_train_ghosting = num_ghosting_artifacts - num_test_ghosting
num_train_non_ghosting = num_non_ghosting_artifacts - num_test_non_ghosting

train_ghosting = ghosting_artifacts[num_test_ghosting:]
test_ghosting = ghosting_artifacts[:num_test_ghosting]

train_non_ghosting = non_ghosting_artifacts[num_test_non_ghosting:]
test_non_ghosting = non_ghosting_artifacts[:num_test_non_ghosting]

train_dataset = train_ghosting + train_non_ghosting
test_dataset = test_ghosting + test_non_ghosting

train_patches, train_labels  = zip(*train_dataset)
test_patches, test_labels = zip(*test_dataset)

train_patches = np.array(train_patches)
train_labels = np.array(train_labels)

print(f" Total Train Patches: {len(train_patches)}")
print(f" Total Train Labels: {len(train_labels)}")

test_patches = np.array(test_patches)
test_labels = np.array(test_labels)

print(f" Total Test Patches: {len(test_patches)}")
print(f" Total Test Labels: {len(test_labels)}")

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

ghosting_patches = train_patches[train_labels == 1]

# ghosting_patches_expanded = np.expand_dims(ghosting_patches, axis=-1)
if ghosting_patches.ndim == 3:
    ghosting_patches_expanded = np.expand_dims(ghosting_patches, axis=-1)
else:
    ghosting_patches_expanded = ghosting_patches


augmented_images = augmented_images(ghosting_patches_expanded, num_augmented_images_per_original=12)

augmented_images_np = np.stack(augmented_images)
augmented_labels = np.ones(len(augmented_images_np))

train_patches_expanded = np.expand_dims(train_patches, axis=-1)
augmented_images_np_expanded = np.expand_dims(augmented_images_np, axis=-1)

train_patches_combined = np.concatenate((train_patches_expanded, augmented_images_np_expanded), axis=0)
train_labels_combined = np.concatenate((train_labels, augmented_labels), axis=0)

print(f" Total Augmented Patches: {len(train_patches_combined)}")
aghosting_patches = train_patches_combined[train_labels_combined == 1]
print(f" Total Augmented GA: {len(aghosting_patches)}")

X_train, X_temp, y_train, y_temp = train_test_split(train_patches_combined, train_labels_combined, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

CX_train = X_train
Cy_train = y_train

print(f"X_Train Shape: {X_train.shape}")
print(f"y_Train Shape: {y_train.shape}")

print(f"X_Val Shape: {X_val.shape}")
print(f"y_Val Shape: {y_val.shape}")

print(f"X_Test Shape: {X_test.shape}")
print(f"y_Test Shape: {y_test.shape}")





opt = Adam(learning_rate=2e-05)
cnn_wcw_model = create_cnn_model()
cnn_wcw_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
wcw_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='/Code/Models/CNN_MultiFeature_wCW_SIGMOID.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1 )
wcw_model_early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, restore_best_weights=True)
wcw_history = cnn_wcw_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[wcw_model_checkpoint, wcw_model_early_stopping])



test_loss, test_acc = cnn_wcw_model.evaluate(X_test, y_test)
test_acc  = test_acc * 100

print(f"Augmented Test Accuracy: {test_acc}")
print(f"Augmented Test Loss: {test_loss}")

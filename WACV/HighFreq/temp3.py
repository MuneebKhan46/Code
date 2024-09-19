import os
import cv2
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate

from sklearn.utils import class_weight
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

denoised_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/denoised'
csv_path = '/ghosting-artifact-metric/Code/WACV/HighFreq/high_frequency_classification_label.csv'

result_file_path = "/ghosting-artifact-metric/Code/WACV/Google_Results.csv"


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



def load_data_from_csv(csv_path, denoised_dir):
  df = pd.read_csv(csv_path)
    
  all_denoised_patches = []
  all_scores = []
  denoised_image_names = []
  all_patch_numbers = []

  for _, row in df.iterrows():
    denoised_file_name = f"denoised_{row['image_name']}.raw"
    denoised_path = os.path.join(denoised_dir, denoised_file_name)

    denoised_patches, denoised_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(denoised_path, row['width'], row['height'])

    all_denoised_patches.extend(denoised_patches)
    denoised_image_names.extend([row['image_name']] * len(denoised_patches))
    all_patch_numbers.extend(denoised_patch_numbers) 
    patch_scores = row['patch_score'].strip('[]').split(', ')
    scores = np.array([0 if float(score) == 0 else 1 for score in patch_scores])
        
    if len(scores) != len(denoised_patches):
      print(f"Error: Mismatch in number of patches and scores for {row['image_name']}")
      continue
    all_scores.extend(scores)
  
  return all_denoised_patches, all_scores, denoised_image_names, all_patch_numbers



def prepare_data(data, labels):
  data = np.array(data).astype('float32') / 255.0
  lbl = np.array(labels)
  return data, lbl

def create_cnn_model(input_shape=(224,224, 1)):
  input_spatial = Input(shape=input_shape)
  input_frequency = Input(shape=input_shape)
  
  x1= Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_spatial)
  x1= Conv2D(32, kernel_size=(3,3), activation='relu')(x1)
  x1= Dropout(0.5)(x1)
  x1= MaxPooling2D(pool_size=(3,3))(x1)
  x1= BatchNormalization()(x1)
  x1= Conv2D(64, kernel_size=(3,3), activation='relu')(x1)
  x1= Conv2D(64, kernel_size=(3,3), activation='relu')(x1)
  x1= Dropout(0.5)(x1)
  x1= Conv2D(128, kernel_size=(3,3), activation='relu')(x1)
  x1= Conv2D(128, kernel_size=(3,3), activation='relu')(x1)
  x1= Dropout(0.5)(x1)
  x1= BatchNormalization()(x1)


  x2= Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_frequency)
  x2= Conv2D(32, kernel_size=(3,3), activation='relu')(x2)
  x2= Dropout(0.5)(x2)
  x2= MaxPooling2D(pool_size=(3,3))(x2)
  x2= BatchNormalization()(x2)
  x2= Conv2D(64, kernel_size=(3,3), activation='relu')(x2)
  x2= Conv2D(64, kernel_size=(3,3), activation='relu')(x2)
  x2= Dropout(0.5)(x2)
  x2= Conv2D(128, kernel_size=(3,3), activation='relu')(x2)
  x2= Conv2D(128, kernel_size=(3,3), activation='relu')(x2)
  x2= Dropout(0.5)(x2)
  x2= BatchNormalization()(x2)
  
  x = concatenate([x1, x2])
  x = Flatten()(x)
  x = Dense(128, activation='elu')(x)
  x = Dropout(0.5)(x)
  output = Dense(1, activation='sigmoid')(x)
  model = Model(inputs=[input_spatial, input_frequency], outputs=output)
  return model
  


def compute_fft_features(patches):
  fft_features = []
  for patch in patches:
    f = np.fft.fft2(patch.squeeze()) 
    fshift = np.fft.fftshift(f)  
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    fft_features.append(magnitude_spectrum.reshape(224, 224, 1))
    
  fft_features_array = np.array(fft_features)
  return fft_features_array


def focal_loss(alpha=0.25, gamma=2.0):
  def focal_loss_fixed(y_true, y_pred):
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.cast(y_pred, K.floatx())
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    CE = -K.log(pt)
    FL = alpha * K.pow(1 - pt, gamma) * CE
    return K.mean(FL)
  return focal_loss_fixed


def combined_loss(y_true, y_pred):
  fl = focal_loss(alpha=0.25, gamma=2.0)(y_true, y_pred)
  ce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
  per_replica_loss = fl + class_weight_ratio * ce
  return tf.reduce_sum(per_replica_loss) * (1.0 / tf.distribute.get_strategy().num_replicas_in_sync)


denoised_patches, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(csv_path, denoised_dir)
fft_patches = compute_fft_features(denoised_patches)

denoised_patches_np, labels_np = prepare_data(denoised_patches, labels)
fft_patches_np, _ = prepare_data(fft_patches, labels)


X_train_spatial, X_temp_spatial, X_train_fft, X_temp_fft, y_train, y_temp = train_test_split( denoised_patches_np, fft_patches_np, labels_np, test_size=0.2, random_state=42)
X_val_spatial, X_test_spatial, X_val_fft, X_test_fft, y_val, y_test = train_test_split(X_temp_spatial, X_temp_fft, y_temp, test_size=0.5, random_state=42)


print(f"Training set: {X_train_spatial.shape}, {X_train_fft.shape}, {y_train.shape}")
print(f"Validation set: {X_val_spatial.shape}, {X_val_fft.shape}, {y_val.shape}")
print(f"Test set: {X_test_spatial.shape}, {X_test_fft.shape}, {y_test.shape}")


class_weights = class_weight.compute_class_weight(
  class_weight='balanced',
  classes=np.unique(y_train),
  y=y_train)

class_weight_dict = dict(enumerate(class_weights))
print(f"Class Weights: {class_weight_dict}")


a=class_weight_dict[1]
print(f"Class Weights ratio: {a}")

class_weight_ratio = 3.14



early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('/ghosting-artifact-metric/Code/WACV/HighFreq/New_Model3.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)


model = create_cnn_model(input_shape=(224, 224, 2))
opt = Adam(learning_rate=2e-05)
model.compile(optimizer=opt, loss=combined_loss, metrics=['accuracy'])
wcw_history = model.fit([X_train_spatial, X_train_fft], y_train, epochs=100, class_weight=class_weight_dict, validation_data=([X_val_spatial, X_val_fft], y_val), callbacks=[checkpoint, reduce_lr, early_stopping])


model = load_model('/ghosting-artifact-metric/Code/WACV/HighFreq/New_Model3.h5',custom_objects={'combined_loss': combined_loss})


test_loss, test_acc = model.evaluate([X_test_spatial, X_test_fft], y_test)
print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

y_pred_prob = model.predict([X_test_spatial, X_test_fft]) 
y_pred = np.where(y_pred_prob > 0.5, 1, 0)


conf_matrix = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()
accuracy_0 = TN / (TN + FP) * 100
accuracy_1 = TP / (TP + FN) * 100
print(f"Non-Ghosting Accuracy: {accuracy_0}")
print(f"Ghosting Accuracy: {accuracy_1}")


class_report = classification_report(y_test, y_pred, target_names=["Non-Ghosting", "Ghosting"], output_dict=True)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Ghosting", "Ghosting"]))

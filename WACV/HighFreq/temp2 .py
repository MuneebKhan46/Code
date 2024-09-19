import os
import cv2
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate

from sklearn.utils import class_weight
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import optuna  # Import Optuna

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

        denoised_patches, denoised_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(
            denoised_path, row['width'], row['height'])

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


def create_cnn_model(input_shape=(224, 224, 2), num_filters=32, dropout_rate=0.5, dense_units=128):
    model = Sequential()
    model.add(Conv2D(num_filters, kernel_size=(3, 3), activation='elu', input_shape=input_shape))
    model.add(Conv2D(num_filters, kernel_size=(3, 3), activation='elu'))

    model.add(Dropout(dropout_rate))

    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(num_filters * 2, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(num_filters * 2, kernel_size=(3, 3), activation='elu'))

    model.add(Dropout(dropout_rate))

    model.add(Conv2D(num_filters * 4, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(num_filters * 4, kernel_size=(3, 3), activation='elu'))

    model.add(Dropout(dropout_rate))

    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(dense_units, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def compute_fft_features(patches):
    fft_features = []
    for patch in patches:
        f = np.fft.fft2(patch.squeeze())
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        fft_features.append(magnitude_spectrum.reshape(224, 224, 1))

    denoised_patches_array = np.array(patches).reshape(-1, 224, 224, 1)
    fft_features_array = np.array(fft_features)
    combined_patches = np.concatenate([denoised_patches_array, fft_features_array], axis=-1)
    return combined_patches


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


# Load and prepare data
denoised_patches, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(csv_path, denoised_dir)
combined_patches = compute_fft_features(denoised_patches)

diff_patches_np, labels_np = prepare_data(combined_patches, labels)

print(f" Total Patches: {len(diff_patches_np)}")
print(f" Patch shape: {diff_patches_np[0].shape}")
print(f" Total Labels: {len(labels_np)}")

X_train, X_temp, y_train, y_temp = train_test_split(diff_patches_np, labels_np, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


def print_class_distribution(y, dataset_name):
    unique, counts = np.unique(y, return_counts=True)
    print(f"{dataset_name} class distribution: {dict(zip(unique, counts))}")


print_class_distribution(y_train, "Training")
print_class_distribution(y_val, "Validation")
print_class_distribution(y_test, "Test")

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train)

class_weight_dict = dict(enumerate(class_weights))
print(f"Class Weights: {class_weight_dict}")

a = class_weight_dict[1]
print(f"Class Weights ratio: {a}")

class_weight_ratio = 3.14

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-5, 1e-3)
    num_filters = trial.suggest_categorical('num_filters', [32, 64, 128])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.3, 0.5, 0.7)
    dense_units = trial.suggest_categorical('dense_units', [64, 128, 256])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])


    model = create_cnn_model(input_shape=(224, 224, 2), num_filters=num_filters, dropout_rate=dropout_rate, 
                             dense_units=dense_units)

    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

  
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=0)

  
    history = model.fit(X_train, y_train, epochs=20, class_weight=class_weight_dict, validation_data=(X_val, y_val), 
                        callbacks=[early_stopping], verbose=0)


    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    trial.report(val_loss, step=0)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()


    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Best hyperparameters: ', study.best_params)

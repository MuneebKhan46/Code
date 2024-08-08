import tensorflow as tf
import numpy as np
import os
from os import path
import csv
import textwrap
import pandas as pd
import resource
from tensorflow.keras.regularizers import l1
from scipy.stats import mode

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, precision_recall_curve
from tensorflow.keras.optimizers import Adam


models = []
class_1_accuracies = []

original_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/original'
denoised_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/denoised'
csv_path     = '/FINAL DATASET/Non_Zeros_Classified_label_filtered.csv'

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

        patch_scores = row['patch_score'].strip('[]').split(', ')
        scores = np.array([0 if float(score) == 0 else 1 for score in patch_scores])
        
        if len(scores) != len(original_patches) or len(scores) != len(denoised_patches):
            print(f"Error: Mismatch in number of patches and scores for {row['image_name']}")
            continue
        
        all_scores.extend(scores)

    return all_original_patches, all_denoised_patches, all_scores, denoised_image_names, all_patch_numbers

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################


def calculate_difference(original, ghosting):
    return [ghost.astype(np.int16) - orig.astype(np.int16) for orig, ghost in zip(original, ghosting)]

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

def prepare_data(data, labels):
    data = np.array(data).astype('float32') / 255.0
    lbl = np.array(labels)
    return data, lbl

#########################################################################################################################################################################################################################################
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
#########################################################################################################################################################################################################################################

def augmented_images(data, num_augmented_images_per_original):
    augmented_images = []
    
    data_augmentation = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.5,
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
#########################################################################################################################################################################################################################################

original_patches, denoised_patches, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(csv_path, original_dir, denoised_dir)

# diff_patches = calculate_difference(original_patches, denoised_patches)
# diff_patches_np, labels_np = prepare_data(diff_patches, labels)


diff_patches_np, labels_np = prepare_data(denoised_patches, labels)


combined = list(zip(diff_patches_np, labels_np, denoised_image_names, all_patch_numbers))
combined = sklearn_shuffle(combined)


ghosting_artifacts = [item for item in combined if item[1] == 1]
non_ghosting_artifacts = [item for item in combined if item[1] == 0]

num_ghosting_artifacts = len(ghosting_artifacts)
num_non_ghosting_artifacts = len(non_ghosting_artifacts)

print(f" Total GA Patches: {num_ghosting_artifacts}")
print(f" Total NGA Patches: {num_non_ghosting_artifacts}")
print(f" Total Patches: {len(labels_np)}")

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

num_test_ghosting = 1250
num_test_non_ghosting = 1250

num_train_ghosting = num_ghosting_artifacts - num_test_ghosting
num_train_non_ghosting = num_non_ghosting_artifacts - num_test_non_ghosting

train_ghosting = ghosting_artifacts[num_test_ghosting:]
test_ghosting = ghosting_artifacts[:num_test_ghosting]

train_non_ghosting = non_ghosting_artifacts[num_test_non_ghosting:]
test_non_ghosting = non_ghosting_artifacts[:num_test_non_ghosting]

train_dataset = train_ghosting + train_non_ghosting
test_dataset = test_ghosting + test_non_ghosting

train_patches, train_labels, train_image_names, train_patch_numbers = zip(*train_dataset)
test_patches, test_labels, test_image_names, test_patch_numbers = zip(*test_dataset)

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
non_ghosting_patches = train_patches[train_labels == 0]

ghosting_patches_expanded = np.expand_dims(ghosting_patches, axis=-1)
augmented_images = augmented_images(ghosting_patches_expanded, num_augmented_images_per_original=7)

augmented_images_np = np.stack(augmented_images)
augmented_labels = np.ones(len(augmented_images_np))

train_patches_expanded = np.expand_dims(train_patches, axis=-1)
augmented_images_np_expanded = np.expand_dims(augmented_images_np, axis=-1)

train_patches_combined = np.concatenate((train_patches_expanded, augmented_images_np_expanded), axis=0)
train_labels_combined = np.concatenate((train_labels, augmented_labels), axis=0)


combined_ghosting_patches = train_patches_combined[train_labels_combined == 1]
combined_non_ghosting_patches = train_patches_combined[train_labels_combined == 0]

print(f" Total GA patches Before Augmentation: {len(ghosting_patches)}")
print(f" Total GA patches After Augmentation: {len(combined_ghosting_patches)}")

print(f" Total NGA patches Before Augmentation: {len(non_ghosting_patches)}")
print(f" Total NGA patches After Augmentation: {len(combined_non_ghosting_patches)}")

print(f" Total Patches: {len(train_patches_combined)}")

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

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

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

# ng = len(train_patches[train_labels == 0])
# ga =  len(train_patches[train_labels == 1])
# total = ng + ga

# imbalance_ratio = ng / ga  
# weight_for_0 = (1 / ng) * (total / 2.0)
# weight_for_1 = (1 / ga) * (total / 2.0)
# class_weight = {0: weight_for_0, 1: weight_for_1}

# print('Weight for class 0 (Non-ghosting): {:.2f}'.format(weight_for_0))
# print('Weight for class 1 (Ghosting): {:.2f}'.format(weight_for_1))




class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(planes, (1, 1), strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, (3, 3), strides=stride, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(inplanes, (1, 1), strides=1, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.LeakyReLU(0.1)

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class CNN_Net(tf.keras.Model):
    def __init__(self):
        super(CNN_Net, self).__init__()
        k = 64
        self.conv_1 = layers.Conv2D(k, (3, 5), strides=(1, 1), padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.layer_1 = BottleNeck(k, k)
        self.layer_2 = BottleNeck(k, k)

        self.conv_2 = layers.Conv2D(k*2, (3, 5), strides=(1, 1), padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.layer_3 = BottleNeck(k*2, k*2)

        self.conv_3 = layers.Conv2D(k*4, (1, 5), strides=(1, 1), padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()

        self.layer_4 = BottleNeck(k*4, k*4)
        self.layer_5 = BottleNeck(k*4, k*4)

        self.conv_4 = layers.Conv2D(k*8, (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.bn4 = layers.BatchNormalization()

        self.layer_6 = BottleNeck(k*8, k*8)

        self.conv_5 = layers.Conv2D(k*4, (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.bn5 = layers.BatchNormalization()

        self.layer_7 = BottleNeck(k*4, k*4)

        self.conv_6 = layers.Conv2D(k*2, (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.bn6 = layers.BatchNormalization()

        self.layer_8 = BottleNeck(k*2, k*2)

        self.conv_7 = layers.Conv2D(k, (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.bn7 = layers.BatchNormalization()

        self.layer_9 = BottleNeck(k, k)

        self.conv_8 = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.sig = layers.Activation('sigmoid')

        self.relu = layers.LeakyReLU(0.1)

    def call(self, x):
        x = self.relu(self.bn1(self.conv_1(x)))
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.relu(self.bn2(self.conv_2(x)))
        x = self.layer_3(x)
        x = self.relu(self.bn3(self.conv_3(x)))
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.relu(self.bn4(self.conv_4(x)))
        x = self.layer_6(x)
        x = self.relu(self.bn5(self.conv_5(x)))
        x = self.layer_7(x)
        x = self.relu(self.bn6(self.conv_6(x)))
        x = self.layer_8(x)
        x = self.relu(self.bn7(self.conv_7(x)))
        x = self.layer_9(x)
        x = self.conv_8(x)
        x = self.sig(x)
        x = x * 255.0

        return x


opt = Adam(learning_rate=2e-6)
model = CNN_Net()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, 224, 224, 1))
# model.summary()
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='/ghosting-artifact-metric/Model/CNN.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1 )
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[model_checkpoint])

# history = cnn_wcw_model.fit(X_train, y_train, epochs=50, class_weight=class_weight, validation_data=(X_val, y_val), callbacks=[wcw_model_checkpoint])

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

def eval (model, test_pat, test_label, model_name, feature_name, technique):
    
    test_loss, test_acc = model.evaluate(test_pat, test_label)
    test_acc  = test_acc * 100
    
    predictions = model.predict(test_pat)
    predicted_labels = np.argmax(predictions, axis=1)
    
    report = classification_report(test_label, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

    conf_matrix = confusion_matrix(test_label, predicted_labels)
    
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TP = conf_matrix[1, 1]
    
    total_class_0 = TN + FP
    total_class_1 = TP + FN
    correctly_predicted_0 = TN
    correctly_predicted_1 = TP
    
    
    accuracy_0 = (TN / total_class_0) * 100
    accuracy_1 = (TP / total_class_1) * 100
    
    precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    
    weighted_precision = (precision_0 * total_class_0 + precision_1 * total_class_1) / (total_class_0 + total_class_1)
    weighted_recall = (recall_0 * total_class_0 + recall_1 * total_class_1) / (total_class_0 + total_class_1)
    
    if weighted_precision + weighted_recall > 0:
        weighted_f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    else:
        weighted_f1_score = 0
    
    weighted_f1_score  = weighted_f1_score*100
    weighted_precision = weighted_precision*100
    weighted_recall    = weighted_recall*100
    
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    
    if macro_precision + macro_recall > 0:
        macro_f1_score = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
    else:
        macro_f1_score = 0
      
    macro_f1_score  = macro_f1_score * 100
    macro_precision = macro_precision * 100
    macro_recall    = macro_recall * 100
    
    
    TP_0 = total_class_0 * recall_0
    TP_1 = total_class_1 * recall_1
    FP_0 = total_class_0 * (1 - precision_0)
    FP_1 = total_class_1 * (1 - precision_1)
    FN_0 = total_class_0 * (1 - recall_0)
    FN_1 = total_class_1 * (1 - recall_1)
    
    micro_precision = (TP_0 + TP_1) / (TP_0 + TP_1 + FP_0 + FP_1)
    micro_recall = (TP_0 + TP_1) / (TP_0 + TP_1 + FN_0 + FN_1)
    
    if micro_precision + micro_recall > 0:
        micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    else:
        micro_f1_score = 0
    
    
    micro_f1_score  = micro_f1_score * 100
    micro_precision = micro_precision * 100
    micro_recall    = micro_recall * 100
    
    print("#############################################################################################################################################################################")
    
    print(f"Accuracy: {test_acc:.2f}% | Weighted Precision: {weighted_precision:.2f}%,  Weighted Recall: {weighted_recall:.2f}%, Weighted F1-score: {weighted_f1_score:.2f}%, Loss: {test_loss:.4f}, N.G.A Accuracy: {accuracy_0:.2f}%, G.A Accuracy: {accuracy_1:.2f}%")
    print(f"Accuracy: {test_acc:.2f}% | Micro Precision: {micro_precision:.2f}%,  Micro Recall: {micro_recall:.2f}%, Micro F1-score: {micro_f1_score:.2f}%, Loss: {test_loss:.4f}, N.G.A Accuracy: {accuracy_0:.2f}%, G.A Accuracy: {accuracy_1:.2f}%")
    print(f"Accuracy: {test_acc:.2f}% | Macro Precision: {macro_precision:.2f}%,  Macro Recall: {macro_recall:.2f}%, Macro F1-score: {macro_f1_score:.2f}%, Loss: {test_loss:.4f}, N.G.A Accuracy: {accuracy_0:.2f}%, G.A Accuracy: {accuracy_1:.2f}%")    
    

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

eval (cnn_wcw_model, X_test, y_test, model_name = "CNN", feature_name = "Absolute Gradient Map", technique = "Baseline")


test_patches = np.array(test_patches)
test_patches = test_patches.reshape((-1, 224, 224, 1))

test_labels = np.array(test_labels)

eval (cnn_wcw_model, test_patches, test_labels, model_name = "CNN", feature_name = "Absolute Gradient Map", technique = "Baseline")


import tensorflow as tf
import numpy as np
import os
from os import path
import csv
import textwrap
import pandas as pd
import resource
from skimage.filters import sobel_h, sobel_v

from tensorflow.keras.regularizers import l1
from skimage.feature import local_binary_pattern
from scipy.stats import mode
from tensorflow import keras
from tensorflow.keras.models import Model
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

def calculate_lbp(patches, P=8, R=1):
  lbp_features = []
  for patch in patches:
    lbp = local_binary_pattern(patch, P, R, method="uniform")
    lbp_features.append(lbp)
  return lbp_features

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################


def prepare_texture_data(original_lbp, denoised_lbp):
  lbp_diff_map = []
  for orig_lbp, denoised_lbp in zip(original_lbp, denoised_lbp):
    lbp_diff = (orig_lbp - denoised_lbp)
    lbp_diff_map.append(lbp_diff)
  
  return lbp_diff_map
    
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
    augmented_images_list = []
    
    data_augmentation = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    for i, patch in enumerate(data):
        if patch.ndim == 2:  # If the patch is 2D, make it 3D
            patch = np.expand_dims(patch, axis=-1)
        patch = np.expand_dims(patch, axis=0)  # Ensure patch has the shape (1, height, width, channels)
        
        temp_generator = data_augmentation.flow(patch, batch_size=1)
        
        for _ in range(num_augmented_images_per_original):
            augmented_image = next(temp_generator)[0]
            augmented_images_list.append(augmented_image)
    return np.array(augmented_images_list)

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################

def create_cnn_model(input_shape=(224,224, 1)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='elu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3,3), activation='elu'))

    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(BatchNormalization())     
    

    model.add(Conv2D(64, kernel_size=(3,3), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='elu'))
    model.add(Conv2D(128, kernel_size=(3,3), activation='elu'))
    model.add(Conv2D(128, kernel_size=(3,3), activation='elu'))

    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='elu'))

    model.add(Dense(1, activation='sigmoid'))
    return model
    

original_patches, denoised_patches, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(csv_path, original_dir, denoised_dir)

original_lbp_features = calculate_lbp(original_patches)
denoised_lbp_features = calculate_lbp(denoised_patches)

print(denoised_lbp_features[0].shape)

diff_patches = prepare_texture_data(original_lbp_features, denoised_lbp_features)

print(diff_patches[0].shape)

diff_patches_np, labels_np = prepare_data(diff_patches, labels)

combined = list(zip(diff_patches_np, labels_np, denoised_image_names, all_patch_numbers))
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

ghosting_patches_expanded = np.expand_dims(ghosting_patches, axis=-1)
augmented_images = augmented_images(ghosting_patches_expanded, num_augmented_images_per_original=12)

augmented_images_np = np.stack(augmented_images)
augmented_labels = np.ones(len(augmented_images_np))

train_patches_expanded = np.expand_dims(train_patches, axis=-1)
# augmented_images_np_expanded = np.expand_dims(augmented_images_np, axis=-1)

print(f"Train Shape: {train_patches_expanded.shape}")
print(f"AUG_Train Shape: {augmented_images_np.shape}")

train_patches_combined = np.concatenate((train_patches_expanded, augmented_images_np), axis=0)
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


#########################################################################################################################################################################################################################################
# Without Class Weight
#########################################################################################################################################################################################################################################

opt = Adam(learning_rate=2e-05)
cnn_wcw_model = create_cnn_model()
cnn_wcw_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
wcw_model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='/ghosting-artifact-metric/Project/Models/CNN_LBP_wCW.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1 )
wcw_model_early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, restore_best_weights=True)
wcw_history = cnn_wcw_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[wcw_model_checkpoint, wcw_model_early_stopping])

#########################################################################################################################################################################################################################################
# With Class Weight
#########################################################################################################################################################################################################################################

ng = len(train_patches[train_labels == 0])
ga =  len(train_patches[train_labels == 1])
total = ng + ga

imbalance_ratio = ng / ga  
weight_for_0 = (1 / ng) * (total / 2.0)
weight_for_1 = (1 / ga) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0 (Non-ghosting): {:.2f}'.format(weight_for_0))
print('Weight for class 1 (Ghosting): {:.2f}'.format(weight_for_1))

opt = Adam(learning_rate=2e-05)
cnn_cw_model = create_cnn_model()
cnn_cw_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

cw_model_checkpoint = ModelCheckpoint(filepath='/ghosting-artifact-metric/Project/Models/CNN_LBP_CW.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1 )
cw_model_early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, restore_best_weights=True)

cw_history = cnn_cw_model.fit(X_train, y_train, epochs=50, class_weight=class_weight, validation_data=(X_val, y_val), callbacks=[cw_model_checkpoint, cw_model_early_stopping])

#########################################################################################################################################################################################################################################
# With Class Balance
#########################################################################################################################################################################################################################################
 
combined = list(zip(CX_train, Cy_train))
combined = sklearn_shuffle(combined)

ghosting_artifacts = [item for item in combined if item[1] == 1]
non_ghosting_artifacts = [item for item in combined if item[1] == 0]

print(f"Ghosting Artifacts: {len(ghosting_artifacts)}")
print(f"Non Ghosting Artifacts: {len(non_ghosting_artifacts)}")

num_ghosting_artifacts = len(ghosting_artifacts)


train_val_ghosting = ghosting_artifacts[:num_ghosting_artifacts]
train_val_non_ghosting = non_ghosting_artifacts[:num_ghosting_artifacts]

cb_train_dataset = train_val_ghosting + train_val_non_ghosting
print(f"Class balance train size {len(cb_train_dataset)}")

cb_train_patches, cb_train_labels = zip(*cb_train_dataset)

cb_train_patches = np.array(cb_train_patches)
cb_train_labels = np.array(cb_train_labels)

opt = Adam(learning_rate=2e-05)
cnn_cb_model = create_cnn_model()
cnn_cb_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


cb_model_checkpoint = ModelCheckpoint(filepath='/ghosting-artifact-metric/Project/Models/CNN_LBP_CB.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1 )
cb_model_early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, restore_best_weights=True)

cb_history = cnn_cb_model.fit(cb_train_patches, cb_train_labels, epochs=50, class_weight=class_weight, validation_data=(X_val, y_val), callbacks=[cb_model_checkpoint, cb_model_early_stopping])

#########################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################
# Testing
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
    print(f"Accuracy: {test_acc:.2f}% | Precision: {micro_precision:.2f}%, Recall: {micro_recall:.2f}%, F1-score: {micro_f1_score:.2f}%, Loss: {test_loss:.4f}, N.G.A Accuracy: {accuracy_0:.2f}%, G.A Accuracy: {accuracy_1:.2f}%")
    save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, macro_precision, macro_recall, macro_f1_score, micro_precision, micro_recall, micro_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)

    class_1_precision = micro_precision
    models.append(model)
    class_1_accuracies.append(class_1_precision)



eval (cnn_wcw_model, X_test, y_test, model_name = "CNN", feature_name = "Local Binary Patterns (LBP)", technique = "Baseline")
eval (cnn_cw_model, X_test, y_test, model_name = "CNN", feature_name = "Local Binary Patterns (LBP)", technique = "Class Weight")
eval (cnn_cb_model, X_test, y_test, model_name = "CNN", feature_name = "Local Binary Patterns (LBP)", technique = "Class Balance")


#########################################################################################################################################################################################################################################
## PRECISION ENSEMBLE 
#########################################################################################################################################################################################################################################

test_patches = np.array(test_patches)
test_patches = test_patches.reshape((-1, 224, 224, 1))

test_labels = np.array(test_labels)

weights = np.array(class_1_accuracies) / np.sum(class_1_accuracies)
csv_file_path = '/ghosting-artifact-metric/Project/Models/CNN_LBP_Weights.csv'
np.savetxt(csv_file_path, weights, delimiter=',')

predictions = np.array([model.predict(test_patches).ravel() for model in models])
weighted_predictions = np.tensordot(weights, predictions, axes=([0], [0]))
predicted_classes = (weighted_predictions > 0.5).astype(int)
true_labels = test_labels.ravel()

test_acc = accuracy_score(true_labels, predicted_classes) * 100
test_loss = log_loss(true_labels, weighted_predictions)

weighted_precision, weighted_recall, weighted_f1_score, _ = precision_recall_fscore_support(true_labels, predicted_classes, average='weighted')

weighted_precision *= 100
weighted_recall *= 100
weighted_f1_score *= 100


macro_precision, macro_recall, macro_f1_score, _ = precision_recall_fscore_support(true_labels, predicted_classes, average='macro')

macro_f1_score  = macro_f1_score * 100
macro_precision = macro_precision * 100
macro_recall    = macro_recall * 100


micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(true_labels, predicted_classes, average='micro')

micro_f1_score  = micro_f1_score * 100
micro_precision = micro_precision * 100
micro_recall    = micro_recall * 100

conf_matrix = confusion_matrix(true_labels, predicted_classes)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

total_class_0 = TN + FN
total_class_1 = TP + FP
accuracy_0 = (TN / total_class_0) * 100 if total_class_0 > 0 else 0
accuracy_1 = (TP / total_class_1) * 100 if total_class_1 > 0 else 0


model_name = "CNN"
feature_name = "Local Binary Patterns (LBP)"
technique = "Precision Ensemble"

save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, macro_precision, macro_recall, macro_f1_score, micro_precision, micro_recall, micro_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)

print("####################################################################################################################################################################################################")
print(f"Accuracy: {test_acc:.2f}% | Precision: {micro_precision:.2f}%, Recall: {micro_recall:.2f}%, F1-score: {micro_f1_score:.2f}%, Loss: {test_loss:.4f}, N.G.A Accuracy: {accuracy_0:.2f}%, G.A Accuracy: {accuracy_1:.2f}%")


misclass_En_csv_path = '/ghosting-artifact-metric/Project/Results/Misclassified_Patches/Precision_Ensemble_CNN_LBP_misclassified_patches.csv'
misclassified_indexes = np.where(predicted_classes != true_labels)[0]

misclassified_data = []
for index in misclassified_indexes:
    denoised_image_name = test_image_names[index]
    patch_number = test_patch_numbers[index]
    true_label = true_labels[index]
    predicted_label = predicted_classes[index]

    probability_non_ghosting = 1 - weighted_predictions[index]
    probability_ghosting = weighted_predictions[index]
    
    misclassified_data.append([
        denoised_image_name, patch_number, true_label, predicted_label,
        probability_non_ghosting, probability_ghosting
    ])

misclassified_df = pd.DataFrame(misclassified_data, columns=[
    'Denoised Image Name', 'Patch Number', 'True Label', 'Predicted Label', 
    'Probability Non-Ghosting', 'Probability Ghosting'
])
misclassified_df.to_csv(misclass_En_csv_path, index=False)


#########################################################################################################################################################################################################################################
## AVERAGE ENSEMBLE 
#########################################################################################################################################################################################################################################

predictions = np.array([model.predict(test_patches).ravel() for model in models])
average_predictions = np.mean(predictions, axis=0)

predicted_classes = (average_predictions > 0.5).astype(int)
true_labels = test_labels.ravel()


test_acc = accuracy_score(true_labels, predicted_classes) * 100
test_loss = log_loss(true_labels, weighted_predictions)
print(test_acc)


weighted_precision, weighted_recall, weighted_f1_score, _ = precision_recall_fscore_support(true_labels, predicted_classes, average='weighted')

weighted_precision *= 100
weighted_recall *= 100
weighted_f1_score *= 100


macro_precision, macro_recall, macro_f1_score, _ = precision_recall_fscore_support(true_labels, predicted_classes, average='macro')

macro_f1_score  = macro_f1_score * 100
macro_precision = macro_precision * 100
macro_recall    = macro_recall * 100


micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(true_labels, predicted_classes, average='micro')

micro_f1_score  = micro_f1_score * 100
micro_precision = micro_precision * 100
micro_recall    = micro_recall * 100

conf_matrix = confusion_matrix(true_labels, predicted_classes)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]


total_class_0 = TN + FN
total_class_1 = TP + FP
accuracy_0 = (TN / total_class_0) * 100 if total_class_0 > 0 else 0
accuracy_1 = (TP / total_class_1) * 100 if total_class_1 > 0 else 0


print("####################################################################################################################################################################################################")
print(f"Accuracy: {test_acc:.2f}% | Precision: {micro_precision:.2f}%, Recall: {micro_recall:.2f}%, F1-score: {micro_f1_score:.2f}%, Loss: {test_loss:.4f}, N.G.A Accuracy: {accuracy_0:.2f}%, G.A Accuracy: {accuracy_1:.2f}%")

model_name = "CNN"
feature_name = "Local Binary Patterns (LBP)"
technique = "Average Ensemble"
save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, macro_precision, macro_recall, macro_f1_score, micro_precision, micro_recall, micro_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)


misclass_En_csv_path = '/ghosting-artifact-metric/Project/Results/Misclassified_Patches/Average_Ensemble_CNN_LBP_misclassified_patches.csv'
misclassified_indexes = np.where(predicted_classes != true_labels)[0]

misclassified_data = []
for index in misclassified_indexes:
    denoised_image_name = test_image_names[index]
    patch_number = test_patch_numbers[index]
    true_label = true_labels[index]
    predicted_label = predicted_classes[index]

    probability_non_ghosting = 1 - average_predictions[index]
    probability_ghosting = average_predictions[index]
    
    misclassified_data.append([
        denoised_image_name, patch_number, true_label, predicted_label,
        probability_non_ghosting, probability_ghosting
    ])

misclassified_df = pd.DataFrame(misclassified_data, columns=[
    'Denoised Image Name', 'Patch Number', 'True Label', 'Predicted Label', 
    'Probability Non-Ghosting', 'Probability Ghosting'
])
misclassified_df.to_csv(misclass_En_csv_path, index=False)



#########################################################################################################################################################################################################################################
## VOTE ENSEMBLE 
#########################################################################################################################################################################################################################################

predictions = []
for model in models:
    pred = model.predict(test_patches)
    pred_class = (pred[:, 0] > 0.5).astype(int)
    predictions.append(pred_class)
predictions = np.array(predictions)

voted_predictions = mode(predictions, axis=0)[0].flatten()

true_labels = test_labels.ravel()

test_acc = accuracy_score(true_labels, voted_predictions) * 100
test_loss = log_loss(true_labels, voted_predictions)

weighted_precision, weighted_recall, weighted_f1_score, _ = precision_recall_fscore_support(true_labels, voted_predictions, average='weighted')

weighted_precision *= 100
weighted_recall *= 100
weighted_f1_score *= 100

macro_precision, macro_recall, macro_f1_score, _ = precision_recall_fscore_support(true_labels, voted_predictions, average='macro')

macro_f1_score  = macro_f1_score * 100
macro_precision = macro_precision * 100
macro_recall    = macro_recall * 100

micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(true_labels, voted_predictions, average='micro')

micro_f1_score  = micro_f1_score * 100
micro_precision = micro_precision * 100
micro_recall    = micro_recall * 100

conf_matrix = confusion_matrix(true_labels, voted_predictions)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

# Class-wise accuracy
total_class_0 = TN + FN
total_class_1 = TP + FP
accuracy_0 = (TN / total_class_0) * 100 if total_class_0 > 0 else 0
accuracy_1 = (TP / total_class_1) * 100 if total_class_1 > 0 else 0

print("####################################################################################################################################################################################################")
print(f"Accuracy: {test_acc:.2f}% | Precision: {micro_precision:.2f}%, Recall: {micro_recall:.2f}%, F1-score: {micro_f1_score:.2f}%, Loss: {test_loss:.4f}, N.G.A Accuracy: {accuracy_0:.2f}%, G.A Accuracy: {accuracy_1:.2f}%")

model_name = "CNN"
feature_name = " Local Binary Patterns (LBP)"
technique = "Vote Based Ensemble"

save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, macro_precision, macro_recall, macro_f1_score, micro_precision, micro_recall, micro_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)

misclass_En_csv_path = '/ghosting-artifact-metric/Project/Results/Misclassified_Patches/Vote_Ensemble_CNN_LBP_misclassified_patches.csv'
misclassified_indexes = np.where(voted_predictions != true_labels)[0]

misclassified_data = []
for index in misclassified_indexes:
    denoised_image_name = test_image_names[index]
    patch_number = test_patch_numbers[index]
    true_label = true_labels[index]
    predicted_label = voted_predictions[index]

    probability_non_ghosting = 1 - (np.sum(predictions[:, index]) / len(models))
    probability_ghosting = np.sum(predictions[:, index]) / len(models)
    
    misclassified_data.append([
        denoised_image_name, patch_number, true_label, predicted_label,
        probability_non_ghosting, probability_ghosting
    ])

misclassified_df = pd.DataFrame(misclassified_data, columns=[
    'Denoised Image Name', 'Patch Number', 'True Label', 'Predicted Label', 
    'Probability Non-Ghosting', 'Probability Ghosting'
])
misclassified_df.to_csv(misclass_En_csv_path, index=False)

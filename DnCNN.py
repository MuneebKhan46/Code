# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# https://github.com/cszn/DnCNN/tree/master/TrainingCodes/dncnn_keras



import os
import glob
import datetime
import numpy as np
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
import keras.backend as K

# Parameters
model_name = 'DnCNN'
batch_size = 128
epochs = 300
initial_lr = 1e-3
save_every = 1
save_dir = os.path.join('models', model_name)
patch_size = 224

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def DnCNN(depth, filters=64, image_channels=3, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None, None, image_channels), name='input' + str(layer_count))
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same', name='conv' + str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu', name='relu' + str(layer_count))(x)

    for i in range(depth - 2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same', use_bias=False, name='conv' + str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn' + str(layer_count))(x)
        layer_count += 1
        x = Activation('relu', name='relu' + str(layer_count))(x)

    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same', use_bias=False, name='conv' + str(layer_count))(x)
    layer_count += 1
    x = Subtract(name='subtract' + str(layer_count))([inpt, x])
    model = Model(inputs=inpt, outputs=x)

    return model


# Learning Rate Schedule
def lr_schedule(epoch):
    if epoch <= 30:
        lr = initial_lr
    elif epoch <= 60:
        lr = initial_lr / 10
    elif epoch <= 80:
        lr = initial_lr / 20
    else:
        lr = initial_lr / 20
    log('Current learning rate is %2.8f' % lr)
    return lr


# Extract patches from an RGB image
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
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)), 'constant')
            patches.append(patch)
            
    return patches 

# Load data from CSV
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

    return np.array(all_original_patches), np.array(all_denoised_patches)



original_dir = '/ghosting-artifact-metric/dataset/dataset_patch_raw_ver3/original'
denoised_dir = '/ghosting-artifact-metric/dataset/dataset_patch_raw_ver3/denoised'
csv_path     = '/ghosting-artifact-metric/dataset/patch_label_median_verified3.csv'
original_patches, denoised_patches = load_data_from_csv(csv_path, original_dir, denoised_dir)



# Data Generator
def data_generator(original_patches, denoised_patches, batch_size):
    num_patches = len(original_patches)
    while True:
        indices = np.arange(num_patches)
        np.random.shuffle(indices)
        for i in range(0, num_patches, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = denoised_patches[batch_indices]
            batch_y = original_patches[batch_indices]
            yield batch_x, batch_y

# Custom Loss Function
def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true)) / 2



model = DnCNN(depth=17, filters=64, image_channels=3, use_bnorm=True)
model.summary()


model.compile(optimizer=Adam(0.001), loss=sum_squared_error)




lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model
history = model.fit(data_generator(original_patches, denoised_patches, batch_size=batch_size),
                    steps_per_epoch=len(original_patches) // batch_size,
                    epochs=epochs,
                    verbose=1,
                    initial_epoch=initial_epoch,
                    callbacks=[lr_scheduler])

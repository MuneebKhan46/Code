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
# =============================================================================


import tensorflow as tf
import os
from PIL import Image
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim



batch_size = 128
epochs = 300
initial_lr = 1e-3
save_every = 1
patch_size = 224
strategy = tf.distribute.MirroredStrategy()


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

def lr_schedule(epoch):
    if epoch <= 30:
        lr = initial_lr
    elif epoch <= 60:
        lr = initial_lr / 10
    elif epoch <= 80:
        lr = initial_lr / 20
    else:
        lr = initial_lr / 20
    print('Current learning rate is %2.8f' % lr)
    return lr

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
    image_array = np.array(image, dtype=np.float32) / 255.0

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

    return np.array(all_original_patches), np.array(all_denoised_patches)

original_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/original'
denoised_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/denoised'
csv_path = '/ghosting-artifact-metric/Code/Non_Zeros_Classified_label_filtered.csv'


original_patches, denoised_patches = load_data_from_csv(csv_path, original_dir, denoised_dir)

train_orig, temp_orig, train_denoised, temp_denoised = train_test_split(original_patches, denoised_patches, test_size=0.2, random_state=42)

val_orig, test_orig, val_denoised, test_denoised = train_test_split(temp_orig, temp_denoised, test_size=0.5, random_state=42)

def data_generator(train_orig, train_denoised, batch_size):
    num_patches = len(train_orig)
    while True:
        indices = np.arange(num_patches)
        np.random.shuffle(indices)
        for i in range(0, num_patches, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = train_denoised[batch_indices]
            batch_y = train_orig[batch_indices]
            yield batch_x, batch_y

def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true)) / 2

with strategy.scope():
    model = DnCNN(depth=17, filters=64, image_channels=3, use_bnorm=True)
    model.compile(optimizer=Adam(0.001), loss=sum_squared_error)
    model_checkpoint = ModelCheckpoint(filepath='/ghosting-artifact-metric/Code/DnCNN.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    lr_scheduler = LearningRateScheduler(lr_schedule)



history = model.fit( data_generator(train_orig, train_denoised, batch_size=batch_size), steps_per_epoch=len(train_orig) // batch_size, epochs=epochs, verbose=1, validation_data=data_generator(val_orig, val_denoised, batch_size=batch_size), 
                    validation_steps=len(val_orig) // batch_size, callbacks=[lr_scheduler, model_checkpoint])


# with tf.keras.utils.custom_object_scope({'sum_squared_error': sum_squared_error}):
#     model = tf.keras.models.load_model("/ghosting-artifact-metric/Code/DnCNN.keras")

predictions = model.predict(test_denoised, batch_size=batch_size)
psnr_values, ssim_values = [], []

for i in range(len(test_orig)):

    psnr_value = psnr(test_orig[i], predictions[i])
    
    patch_size = min(test_orig[i].shape[0], test_orig[i].shape[1])
    win_size = min(7, patch_size)
    
    if win_size >= 3:
        ssim_value = ssim(test_orig[i], predictions[i], win_size=win_size, channel_axis=-1, data_range=1.0)
        ssim_values.append(ssim_value)
    else:
        print(f"Skipping SSIM for image {i} due to insufficient size (patch size: {patch_size})")
    
    psnr_values.append(psnr_value)


avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values) if ssim_values else 0

print(f"Average PSNR: {avg_psnr:.4f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")

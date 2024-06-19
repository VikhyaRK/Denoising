#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import load_model
import glob
import imageio
import numpy as np
import os

# PSNR calculation
def calculate_psnr(true_image, predicted_image):
    return tf.image.psnr(true_image, predicted_image, max_val=1.0)

# Load pre-trained model with custom PSNR metric
model = load_model('Models/mymodel2.h5', custom_objects={'psnr_metric': calculate_psnr})

# Recursive image enhancement function
def enhance_image(image, iterations, flag):
    if iterations == 0:
        return image

    h, w, c = image.shape
    if flag == 1:
        predicted = model.predict(image.reshape(1, h, w, 3))
        normalized_image = image / 255.0
        enhanced = normalized_image + ((predicted[0] * normalized_image) * (1 - normalized_image))
        psnr_value = calculate_psnr(tf.convert_to_tensor(image, dtype=tf.float32), tf.convert_to_tensor(enhanced * 255, dtype=tf.float32)).numpy()
        print(f"PSNR: {psnr_value:.4f}")
        return enhance_image(enhanced, iterations - 1, 0)
    else:
        predicted = model.predict(image.reshape(1, h, w, 3))
        enhanced = image + ((predicted[0] * image) * (1 - image))
        psnr_value = calculate_psnr(tf.convert_to_tensor(image, dtype=tf.float32), tf.convert_to_tensor(enhanced, dtype=tf.float32)).numpy()
        print(f"PSNR: {psnr_value:.4f}")
        return enhance_image(enhanced, iterations - 1, flag)

# Load images from the specified directory
input_path = 'test/low'
image_files = glob.glob(input_path + "/*")
image_list = []

image_files.sort()
for filename in image_files:
    image = imageio.imread(filename)
    image_list.append(image)
images = np.array(image_list)

# Enhance the first image
input_image = images[0]
enhanced_img = enhance_image(input_image, 8, 1)

# Save the enhanced image to the specified directory
output_dir = 'test/predicted'
os.makedirs(output_dir, exist_ok=True)

output_file_path = os.path.join(output_dir, 'enhanced_image.png')
imageio.imwrite(output_file_path, (enhanced_img * 255).astype(np.uint8))
print(f"Enhanced image saved to {output_file_path}")


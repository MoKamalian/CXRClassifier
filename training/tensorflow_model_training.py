"""
@author amir Kamalian
@date   13 nov 2023

This is the script used to train the TensorFlow model used for classification. The images
used are in chest_xray/train.

"""

import os
import tensorflow
import cv2
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator


# image resizing
images_NORMAL = '../chest_xray/train/NORMAL'
images_PNEUMONIA = '../chest_xray/train/PNEUMONIA'

output_NORMAL = 'preprocessed_images/NORMAL'
output_PNEUMONIA = 'preprocessed_images/PNEUMONIA'

size = (256, 256)

""" 
def resize_images(input_path: str, output_path: str, resize_to: (int, int)) -> None:
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in os.listdir(input_path):
        if file.endswith(".jpeg") or file.endswith(".png"):
            img_path = os.path.join(input_path, file)
            image = cv2.imread(img_path)

            # image resized while preserving aspect ratio
            img_resized = cv2.resize(image, resize_to, interpolation=cv2.INTER_AREA)

            # save images to a new path
            output = os.path.join(output_path, file)
            cv2.imwrite(output, img_resized)


# resize NORMAL
resize_images(images_NORMAL, output_NORMAL, size)

# resize PNEUMONIA
resize_images(images_PNEUMONIA, output_PNEUMONIA, size)
"""

# path to training images (i.e. chest_xray/train/NORMAL and chest_xray/train/PNEUMONIA)
training_images = 'preprocessed_images'
batch_size = 32


# data normalization
train_set = ImageDataGenerator(rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

generate_set = train_set.flow_from_directory(training_images, target_size=size, batch_size=32, class_mode='binary')

# CNN definition
cls_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size[0], size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# model compilation
cls_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train using the images training set
cls_model.fit(generate_set, epochs=10)

# saving model to be using with openCV
cls_model.save('trained_model')



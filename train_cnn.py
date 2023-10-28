# -*- coding: utf-8 -*-
# @Time    : 2023/10/27
# @Author  : Lei
# @Email   : 6222ppt@gmail.com
# @File    : train_cnn.py
# @Software: PyCharm
# @Brief   : CNN model training code, the training code will be saved in the 'models' directory, and the line chart will be saved in the 'results' directory.

import tensorflow as tf
import matplotlib.pyplot as plt
from time import *


# Dataset loading function, specify the dataset's location and uniformly process to a size of imgheight*imgwidth, while also setting the batch.
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # Load the training set.
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # Load the testing set.
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # Return the processed training set, validation set, and class names.
    return train_ds, val_ds, class_names


# Construct the CNN model.
def model_load(IMG_SHAPE=(224, 224, 3), class_num=12):
    # Build the model.
    model = tf.keras.models.Sequential([
        # Normalize the model, converting numbers between 0-255 to be between 0 and 1.
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        # Convolutional layer, the output of this layer has 32 channels, the size of the convolutional kernel is 3*3, and the activation function is relu.
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # Add a pooling layer, with a kernel size of 2*2.
        tf.keras.layers.MaxPooling2D(2, 2),
        # Add another convolution
        # Convolutional layer, the output has 64 channels, the size of the convolutional kernel is 3*3, and the activation function is relu.
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # Pooling layer, max pooling, pooling operation over a 2*2 area.
        tf.keras.layers.MaxPooling2D(2, 2),
        # Convert the two-dimensional output to one-dimensional.
        tf.keras.layers.Flatten(),
        # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
        tf.keras.layers.Dense(128, activation='relu'),
        # Use the softmax function to output the model to neurons of the length of the class names, and the activation function adopts softmax for corresponding probability values.
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # Output model information.
    model.summary()
    # Specify the training parameters of the model, with the optimizer being the SGD optimizer, and the loss function being the cross-entropy loss function.
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    # Return the model.
    return model


# Display the training process curve.
def show_loss_acc(history):
    # Extract training and validation accuracy information and error information from the history.
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Output the picture in a top-down structure.
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results/results_cnn.png', dpi=100)


def train(epochs):
    # Start training, and record the start time.
    begin_time = time()
    # todo Load the dataset, modify the path to your dataset.
    train_ds, val_ds, class_names = data_load("../data/vegetable_fruit/image_data",
                                              "../data/vegetable_fruit/test_image_data", 224, 224, 16)
    print(class_names)
    # Load the model.
    model = model_load(class_num=len(class_names))
    # Specify the number of training epochs and then start training.
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # todo Save the model, modify it to the name of the model you want to save.
    model.save("models/cnn_fv.h5")
    # Record the end time.
    end_time = time()
    run_time = end_time - begin_time
    print('The runtime of this loop program:', run_time, "s")  # The runtime of this loop program: 1.4201874732
    # Create a model training process chart.
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=30)

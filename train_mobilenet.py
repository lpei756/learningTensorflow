# -*- coding: utf-8 -*-
# @Time    : 2023/10/27
# @Author  : Lei
# @Email   : 6222ppt@gmail.com
# @File    : train_mobilenet.py
# @Software: PyCharm
# @Brief   : MobileNet model training code, the trained model will be saved in the 'models' directory, and the line chart will be saved in the 'results' directory.

import tensorflow as tf
import matplotlib.pyplot as plt
from time import *


# Dataset loading function, specify the location of the dataset, resize it uniformly to imgheight*imgwidth dimensions, and set the batch size.
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # Load the training dataset.
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # Load the testing dataset.
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # Return the processed training dataset, validation dataset, and class names.
    return train_ds, val_ds, class_names


# Build a MobileNet model.
# Model loading, specify image processing size, and whether to perform transfer learning.
def model_load(IMG_SHAPE=(224, 224, 3), class_num=12):
    # Normalization is not needed during fine-tuning.
    # Load a pre-trained MobileNet model.
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    # Freeze the model's backbone parameters.
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        # Perform normalization.
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
        # Set the backbone model.
        base_model,
        # Perform global average pooling on the output of the backbone model.
        tf.keras.layers.GlobalAveragePooling2D(),
        # Map it to the final number of classes through a fully connected layer.
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
    # The model is trained with the Adam optimizer and the categorical cross-entropy loss function.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Display the training process curves.
def show_loss_acc(history):
    # Extract model training and validation accuracy and loss information from the history.
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
    plt.savefig('results/results_mobilenet.png', dpi=100)


def train(epochs):
    # Start training and record the start time.
    begin_time = time()
    # todo Load the dataset, modify it to the path of your dataset.
    train_ds, val_ds, class_names = data_load("../data/vegetable_fruit/image_data",
                                              "../data/vegetable_fruit/test_image_data", 224, 224, 16)
    print(class_names)
    # Load the model.
    model = model_load(class_num=len(class_names))
    # Specify the number of training epochs and start training.
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # todo Save the model, modify it to the name of the model you want to save.
    model.save("models/mobilenet_fv.h5")
    # Record the end time.
    end_time = time()
    run_time = end_time - begin_time
    print('The runtime of this loop program:', run_time, "s")  # The loop program runs time: 1.4201874732
    # Draw a model training process chart.
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=30)

import tensorflow as tf
import matplotlib.pyplot as plt
from time import *


# Load data set
# def data_load(data_dir, img_height, img_width, batch_size):
#     train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#         data_dir,
#         label_mode='categorical',
#         validation_split=0.2,
#         subset="training",
#         seed=123,
#         color_mode="rgb",
#         image_size=(img_height, img_width),
#         batch_size=batch_size)
#
#     val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#         data_dir,
#         label_mode='categorical',
#         validation_split=0.2,
#         subset="validation",
#         seed=123,
#         color_mode="rgb",
#         image_size=(img_height, img_width),
#         batch_size=batch_size)
#
#     class_names = train_ds.class_names
#
#     return train_ds, val_ds, class_names
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


# Model loading, specifying the size of the image processing and whether transfer learning is performed
def model_load(IMG_SHAPE=(224, 224, 3), class_num=245):
    # No normalization is required in the process of fine tuning
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
    # Model training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Show the curve of the training process
def show_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

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
    begin_time = time()
    train_ds, val_ds, class_names = data_load("../../data/Fruit-Images-Dataset/Training", "../data/Fruit-Images-Dataset/Test", 224, 224, 16)
    print(class_names)
    model = model_load(class_num=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("models/mobilenet_fruits.h5")
    end_time = time()
    run_time = end_time - begin_time
    print('The loop program runs time:', run_time, "s")  # The loop program runs time: 1.4201874732
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=10)
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.regularizers import l2

from Check import show_random_examples

# Create 'models' directory if it doesn't exist
if not os.path.isdir('models'):
    os.mkdir('models')

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define batch size and number of epochs
batch_size = 128
epochs = 100

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,  # Reduced rotation range
    width_shift_range=0.1,  # Reduced width shift range
    height_shift_range=0.1,  # Reduced height shift range
    horizontal_flip=True
)

# Fit the data generator to the training data
datagen.fit(x_train)


# Define the CNN model
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))

    # Convolutional Block Function
    def add_conv_block(model, num_filters):
        model.add(tf.keras.layers.Conv2D(num_filters, 3, activation='relu', padding='same',
                                         kernel_regularizer=l2(0.01)))  # Added L2 regularization
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(num_filters, 3, activation='relu', padding='valid',
                                         kernel_regularizer=l2(0.01)))  # Added L2 regularization
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))  # Increased dropout rate
        return model

    # Add convolutional blocks
    model = add_conv_block(model, 64)  # Increased filters in first block
    model = add_conv_block(model, 128)
    model = add_conv_block(model, 256)  # Added another convolutional block

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu',
                                    kernel_regularizer=l2(0.01)))  # Added a dense layer with L2 regularization
    model.add(tf.keras.layers.Dropout(0.5))  # Increased dropout rate
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Create and summarize the model
model = create_model()
model.summary()

# Train the model using the data generator
h = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_test / 255., y_test),
    epochs=epochs,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3),  # Increased patience
        tf.keras.callbacks.ModelCheckpoint('models/model_{val_accuracy:.3f}.h5.keras', save_best_only=True,
                                           save_weights_only=False, monitor='val_accuracy')
    ]
)

# Plot training and validation metrics
losses = h.history['loss']
accs = h.history['accuracy']
val_losses = h.history['val_loss']
val_accs = h.history['val_accuracy']
epochs = len(losses)

plt.figure(figsize=(12, 4))
for i, metrics in enumerate(zip([losses, accs], [val_losses, val_accs], ['Loss', 'Accuracy'])):
    plt.subplot(1, 2, i + 1)
    plt.plot(range(epochs), metrics[0], label='Training {}'.format(metrics[2]))
    plt.plot(range(epochs), metrics[1], label='Validation {}'.format(metrics[2]))
    plt.legend()
plt.show()

# Load the best model and make predictions
model = tf.keras.models.load_model('models/model_0.829.h5.keras')
preds = model.predict(x_test / 255.)
show_random_examples(x_test, y_test, preds)
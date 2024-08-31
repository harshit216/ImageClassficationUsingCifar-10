import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50

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
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,  # Added zoom augmentation
    fill_mode='nearest'
)

# Fit the data generator to the training data
datagen.fit(x_train)

# Define the CNN model using Transfer Learning with ResNet50
def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Freeze the base model layers
    base_model.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
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
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),  # Increased patience
        tf.keras.callbacks.ModelCheckpoint('models/model_{val_accuracy:.3f}.h5.keras', save_best_only=True,
                                           save_weights_only=False, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)  # Learning rate scheduler
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
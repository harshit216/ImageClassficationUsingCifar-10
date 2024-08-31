from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('models/model_0.840.h5.keras')

# Class names for CIFAR-10 dataset
class_names = ['aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Load and preprocess the image
    img = Image.open(file)
    img = img.resize((32, 32))  # Resize to match the input shape of the model
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    return f'The predicted class is: {predicted_class}'


if __name__ == '__main__':
    app.run(debug=True)
from glob import glob
from PIL import Image
import random
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
import matplotlib.image as mpimg
from skimage.filters import prewitt_h, prewitt_v

import os
import numpy as np
random.seed(42)

class Data:
    def __init__(self):
        self.image_list = []
        self.labels = []

    def load_images(self, pattern, label):
        files = glob(pattern)

        for file in files:
            im = Image.open(file)
            self.image_list.append(im)
            self.labels.append(label)

        random.shuffle(self.image_list)
        random.shuffle(self.labels)

    def get_data_and_labels(self):
        if not self.image_list or not self.labels:
            # neutral are 0, happy are 1
            self.load_images("../data/*a.jpg", 0)
            self.load_images("../data/*b.jpg", 1)

        # Convert image list and labels to numpy arrays
        self.image_list = np.array(self.image_list)
        self.labels = np.array(self.labels)

        return self.image_list, self.labels


data = Data()
data.get_data_and_labels()
print(data.image_list, data.labels)


class FeatureExtractor:
    def __init__(self, data):
        self.data = data

    def edge_creator(self, images):
        edges_horizontal = []
        edges_vertical = []
        for image in images:
            # Convert PIL Image to numpy array
            image_array = np.array(image)
            # Calculating horizontal edges using Prewitt kernel
            edges_prewitt_horizontal = prewitt_h(image_array)
            # Calculating vertical edges using Prewitt kernel
            edges_prewitt_vertical = prewitt_v(image_array)
            edges_horizontal.append(edges_prewitt_horizontal)
            edges_vertical.append(edges_prewitt_vertical)
        return edges_horizontal, edges_vertical



class ModelBuilder:
    def __init__(self):
        self.model = None
    
    def build_model(self):
        # Load the pre-trained ResNet50 model without including the top (fully connected) layers
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the layers of the pre-trained model
        for layer in base_model.layers:
            layer.trainable = False

        # Add your own classification layers on top of the base model
        x = layers.Flatten()(base_model.output)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation='sigmoid')(x)  # Assuming binary classification

        # Create the model
        self.model = Model(inputs=base_model.input, outputs=x)
        
        return self.model

    def compile_model(self):
        if self.model is None:
            self.build_model()
        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Create an instance of the Data class
data = Data()

# Load and preprocess the data
train_data_neutral, test_data_neutral, train_data_happy, test_data_happy = data.get_train_and_test()

# Convert images to numpy arrays and preprocess them (resize, rescale, etc.)
# Define your preprocessing steps based on the requirements of the ResNet50 model

# Create an instance of the ModelBuilder class
model_builder = ModelBuilder()

# Build and compile the ResNet50 model
model_builder.build_model()
model_builder.compile_model()

# Train the model using your training data
history = model_builder.model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

# Evaluate the model on your test data
test_loss, test_acc = model_builder.model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)

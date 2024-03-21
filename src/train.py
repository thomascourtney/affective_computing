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
import csv
# set seed
random.seed(42)



import os

class Data:
    def __init__(self, pattern, directory):
        """
        Input
        ----------
        - pattern: string of where to get the annotations.csv file from (e.g. "../data/annotations.csv")
        - directory: string of where to get the image data from (e.g. "../data")
        
        """

        self.image_list = []
        self.labels = []
        self.pattern = pattern
        self.directory = directory

    def load_data_from_csv(self, csv_file):
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Assuming the first column contains the file names and the second column contains the labels
                file_name, label = row[0], row[1]
                file_path = os.path.join(self.directory, file_name)  # Construct the file path
                im = Image.open(file_path)
                im_array = np.array(im)  # Convert PIL Image to NumPy array
                self.image_list.append(im_array)
                self.labels.append(label)

    def get_data_and_labels(self):
        if not self.image_list or not self.labels:
            self.load_data_from_csv(self.pattern)

        # Convert labels to numpy array
        self.labels = np.array(self.labels)
        random.shuffle(self.image_list)
        random.shuffle(self.labels)
        return self.image_list, self.labels

data = Data("../data/annotations.csv", "../data")
image_list, labels = data.get_data_and_labels()


train_len = int(len(image_list) * 0.8)
train_data, test_data = image_list[:train_len], image_list[train_len:]
train_labels, test_labels = labels[:train_len], labels[train_len:]

print(len(test_labels))





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



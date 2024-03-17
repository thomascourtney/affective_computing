import glob
from PIL import Image
import random
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.image as mpimg
from skimage.filters import prewitt_h,prewitt_v

class Data:
    def __init__(self):
        self.image_list_neutral = []
        self.image_list_happy = []
        self.train_data_neutral = None
        self.test_data_neutral = None
        self.train_data_happy = None
        self.test_data_happy = None

    def load_images(self, pattern, image_list):
        files = glob.glob(pattern)

        for file in files:
            im = Image.open(file)
            image_list.append(im)

        random.shuffle(image_list)

    def get_images(self):
        self.load_images("../data/*a.jpg", self.image_list_neutral)
        self.load_images("../data/*b.jpg", self.image_list_happy)

        return self.image_list_neutral, self.image_list_happy

    def get_train_and_test(self):
        if not self.image_list_neutral or not self.image_list_happy:
            self.get_images()

        n = len(self.image_list_neutral)
        train_data_len = int(n * 0.8)

        self.train_data_neutral = self.image_list_neutral[:train_data_len]
        self.test_data_neutral = self.image_list_neutral[train_data_len:]
        self.train_data_happy = self.image_list_happy[:train_data_len]
        self.test_data_happy = self.image_list_happy[train_data_len:]

        return (
            self.train_data_neutral,
            self.test_data_neutral,
            self.train_data_happy,
            self.test_data_happy
        )


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

data = Data()
feature_extractor = FeatureExtractor(data)
train_neutral, test_neutral, train_happy, test_happy = data.get_train_and_test()
print(len(test_neutral))

# Extracting edges for the neutral and happy images
neutral_horizontal_edges, neutral_vertical_edges = feature_extractor.edge_creator(train_neutral)
happy_horizontal_edges, happy_vertical_edges = feature_extractor.edge_creator(train_happy)
print(happy_horizontal_edges)
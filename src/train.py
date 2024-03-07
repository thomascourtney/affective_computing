import glob
from PIL import Image
import random
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.image as mpimg


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


data = Data()
train_netrual, test_netural, train_happy, test_happy = data.get_train_and_test()
print(len(test_netural))


def plot():
    # Set up matplotlib fig, and size it to fit 4x4 pics
    nrows = 4
    ncols = 4

    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)
    pic_index = 100
    train_cat_fnames = os.listdir( train_netrual )
    train_dog_fnames = os.listdir( train_dogs_dir )


    next_cat_pix = [os.path.join(train_cats_dir, fname) 
                    for fname in train_cat_fnames[ pic_index-8:pic_index] 
                ]

    next_dog_pix = [os.path.join(train_dogs_dir, fname) 
                    for fname in train_dog_fnames[ pic_index-8:pic_index]
                ]

    for i, img_path in enumerate(next_cat_pix+next_dog_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

    plt.show()
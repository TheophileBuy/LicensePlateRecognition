from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

width = 75
height = 100
channel = 1

def load_data():
        images = np.array([]).reshape(0,height,width)
        labels = np.array([])
        
        ################ Data in  ./AUG then in a folder with label name, example : ./AUG/A for A images #############
        directories = [x[0] for x in os.walk('AUG')][1:]
        print(directories)
        for directory in directories:
                filelist = glob.glob(directory+'/*.jpg')
                sub_images = np.array([np.array(Image.open(fname)) for fname in filelist])
                sub_labels = [int(directory[-2:])]*len(sub_images)
                images = np.append(images,sub_images, axis = 0)
                labels = np.append(labels,sub_labels, axis = 0)
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)
        return (X_train, y_train), (X_test, y_test)

(train_images, train_labels), (test_images, test_labels) = load_data()
train_images = train_images.reshape((train_images.shape[0], height, width, channel))
test_images = test_images.reshape((test_images.shape[0], height, width,channel))
train_images, test_images = train_images / 255.0, test_images / 255.0
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channel)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(35, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=8)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
model.save("model_char_recognition.h5")


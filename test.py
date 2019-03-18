import numpy as np
import os
import pandas as pd
# %matplotlib inline
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils import to_categorical
K.set_image_dim_ordering('th')
from sklearn.preprocessing import MinMaxScaler

cat = np.load('cat.npy')
penguin = np.load('penguin.npy')
ant = np.load('ant.npy')
bee = np.load('bee.npy')
flamingo = np.load('flamingo.npy')
owl = np.load('owl.npy')
pig = np.load('pig.npy')
dolphin = np.load('dolphin.npy')
snake = np.load('snake.npy')
ice_cream = np.load('ice_cream.npy')
sun = np.load('sun.npy')
mushroom = np.load('mushroom.npy')
flower = np.load('flower.npy')
cactus = np.load('cactus.npy')

# print number of images in dataset and numpy array size of each image
print ("no_of_pics, pixels_size")
print(cat.shape)
print(penguin.shape)
print(ant.shape)
print(bee.shape)

cat = np.c_[cat, np.zeros(len(cat))]
penguin = np.c_[penguin, np.ones(len(penguin))]
ant = np.c_[ant, 2*np.ones(len(ant))]
bee = np.c_[bee, 3*np.ones(len(bee))]
flamingo = np.c_[flamingo, 4*np.ones(len(flamingo))]
owl = np.c_[owl, 5*np.ones(len(owl))]
pig = np.c_[pig, 6*np.ones(len(pig))]
dolphin = np.c_[dolphin, 7*np.ones(len(dolphin))]
snake = np.c_[snake, 8*np.ones(len(snake))]
ice_cream = np.c_[ice_cream, 9*np.ones(len(ice_cream))]
sun = np.c_[sun, 10*np.ones(len(sun))]
mushroom = np.c_[mushroom, 11*np.ones(len(mushroom))]
flower = np.c_[flower, 12*np.ones(len(flower))]
cactus = np.c_[cactus, 13*np.ones(len(cactus))]

X = np.concatenate((cat[:10000,:-1], penguin[:10000,:-1], ant[:10000,:-1], bee[:10000,:-1], flamingo[:10000,:-1], owl[:10000,:-1], pig[:10000,:-1], dolphin[:10000,:-1], snake[:10000,:-1]\
                   , ice_cream[:10000,:-1], sun[:10000,:-1], mushroom[:10000,:-1], flower[:10000,:-1], cactus[:10000,:-1]), axis=0).astype('float32') # all columns but the last

y = np.concatenate((cat[:10000,-1], penguin[:10000,-1], ant[:10000,-1], bee[:10000,-1],\
                   flamingo[:10000,-1], owl[:10000,-1], pig[:10000,-1], dolphin[:10000,-1], snake[:10000,-1],\
                    ice_cream[:10000,-1], sun[:10000,-1], mushroom[:10000,-1], flower[:10000,-1], cactus[:10000,-1]\
                   ), axis=0).astype('float32') # the last column

# train/test split (divide by 255 to obtain normalized values between 0 and 1)
# Use a 50:50 split, training the models on 10'000 samples and thus have plenty of samples to spare for testing.
X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.5,random_state=0)

# one hot encode outputs
y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]

# reshape to be [samples][pixels][width][height]
X_train_cnn = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
s = X_train_cnn.shape
print (s, num_classes)

#build a model
model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
    
model.add(Dropout(0.2))
model.add(Flatten())
    
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#compile the model
model.compile(loss="categorical_crossentropy",
             optimizer ="adam", metrics=["accuracy"])

model.summary()

model.fit(
    X_train_cnn,
    y_train_cnn,
    validation_data=(X_test_cnn, y_test_cnn),
    epochs = 30,
    batch_size = 50
)

model.save_weights('quickdraw_neuralnet.h5')
model.save('quickdraw.h5')
model.save('quickdraw.model')
print ("Model is saved")

#accuaracy
model_loss, model_accuracy = model.evaluate(X_test_cnn, y_test_cnn, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


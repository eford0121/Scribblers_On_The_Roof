#  import all the nececessary machine learning library and function
#  scientific computing library for saving, reading, and resizing images
from scipy.misc.pilutil import imsave, imread, imresize
from flask import Flask, render_template,request
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import base64
import re
import cv2 as cv
import sys
import os
from os import walk, getcwd
# add system path relative to this flask app
sys.path.append(os.path.abspath("./model"))
from model import *

app = Flask(__name__)
global model
# initializing the model
model, graph = init()

# create a path to the data folder: numpy image files
Training_data_path = "data/"
File_Name_list = []
for (dirpath, dirnames, filenames) in walk(Training_data_path):
    if filenames != '.DS_Store':
        File_Name_list.extend(filenames)
        break
# implement python built in function : enumerate to loop through the image items, them use the index to loop through the remaining element.
# we are not changing the list we are simply modifying the objects in the list(changind data file type from .npy to png)
# converting image to string
for ndx, member in enumerate(File_Name_list):
    File_Name_list[ndx] = File_Name_list[ndx].replace('.npy', '')

# create an image by using the  string
def convertImage(ImageData):
    Image_string = re.search(b'base64,(.*)',ImageData).group(1)
    with open('output.png','wb') as output:
      output.write(base64.b64decode(Image_string))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():

# get data from drawing canvas and save as image
    Draw_Image = request.get_data()
    convertImage(Draw_Image)
    print("debug")
#  # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.png', mode='L')

    x = preprocess(x)

    x = imresize(x, (32, 32))

    x = x.astype('float32')
    x /= 255

    x = x.reshape(1, 32, 32, 1)
    print("debug2")

    with graph.as_default():

        out = model.predict(x)
        print(out)
# Returns the indices of the maximum values along an axis.
        print(np.argmax(out, axis=1))
        index = np.array(np.argmax(out, axis=1))
        index = index[0]
        sketch = File_Name_list[index]
        print("debug3")
        return sketch

@app.route('/getNoteText',methods=['GET','POST'])
def GetNoteText():
    if request.method == 'POST':
        file = request.files['pic']
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        processImage(filename)
        return "Success"
    else:
        return "No Success"

# Gamma correction and the Power Law Transform
# image pixel intensities must be scaled from the range [0, 255] to [0, 1.0].
def adjust_gamma(image, gamma=1.5):
# build a lookup table mapping the pixel values [0, 255] to  their adjusted gamma values
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
# apply gamma correction using the lookup table
   return cv.LUT(image, table)


def preprocess(img):
  
# Adaptive Thresholding
# threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY_INV, 11, 2)
    return th3

if __name__ == "__main__":
	
	app.run(debug=True)  
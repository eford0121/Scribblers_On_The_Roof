import os
from flask import Flask, request, jsonify, render_template

import keras
from keras.preprocessing import image
from keras import backend as K

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
graph = None

# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
def load_model():
    global model
    global graph
    model = keras.models.load_model("quickdraw.h5")
    graph = K.get_session().graph


load_model()


def prepare_image(img):
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    # Scale from 0 to 255
    img /= 255
    # Invert the pixels
    img = 1 - img
    # Flatten the image to an array of pixels
    image_array = img.flatten().reshape(-1, 28 * 28)
    image_array = img.flatten().reshape(1,1,28,28)

    # Return the processed feature array
    return image_array
    
@app.route('/')
def index():
    data = "mushroom"
    return render_template("index.html", data=data)

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        print(request)

        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(filepath)

            # Load the saved image using Keras and resize it to the mnist
            # format of 28x28 pixels
            image_size = (28, 28)
            im = image.load_img(filepath, target_size=image_size,
                                grayscale=True)

            # Convert the 2D image to an array of pixel values
            image_array = prepare_image(im)
            print(image_array)

            # Get the tensorflow default graph and use it to make predictions
            global graph
            with graph.as_default():

                # Use the model to make a prediction
                predicted_digit = model.predict_classes(image_array)[0]
                
                if predicted_digit == 0:
                    predicted_image = "CAT"
                elif predicted_digit == [1]:
                    predicted_image = "PENGIUN"
                elif predicted_digit == [2]:
                    predicted_image = "ANT"  
                elif predicted_digit == [3]:
                    predicted_image = "BEE"
                elif predicted_digit == [4]:
                    predicted_image = "FLAMINGO"
                elif predicted_digit == [5]:
                    predicted_image = "OWL" 
                elif predicted_digit == [6]:
                    predicted_image = "PIG"
                elif predicted_digit == [7]:
                    predicted_image = "DOLPHIN"
                elif predicted_digit == [8]:
                    predicted_image = "SNAKE"
                elif predicted_digit == [9]:
                    predicted_image = "ICE CREAM"
                elif predicted_digit == [10]:
                    predicted_image = "SUN"
                elif predicted_digit == [11]:
                    predicted_image = "MUSHROOM"
                elif predicted_digit == [12]: 
                    predicted_image = "FLOWER"
                elif predicted_digit == [13]:
                    predicted_image = "CACTUS"
                else: 
                    predicted_image = "Cannot Recognize Your Scribble"
                
                # Indicates the Predicted Index Number
                data["predicted_digit"] = str(predicted_digit)
                # Indicates the Predicted Image
                data["predicted_image"] = str(predicted_image)
                # indicate that the request was a success
                data["success"] = True

                #data = jsonify(data)

            return render_template('index.html', data=data)
            
            #return jsonify(data)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run(debug=True,port=8008)
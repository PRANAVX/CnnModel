import base64
import cv2
import io
import json
from flask import Flask
from flask import jsonify
from flask import request,render_template
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

app=Flask(__name__)

def get_model():
    global model
    model=load_model("Faceshape_predictor_app.h5")
    print("model loaded")


def preprocess_image(image):
    image = np.asarray(image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(64,64))
	
    image = image[np.newaxis,...]
    image = image[...,np.newaxis]
    return image

get_model()
@app.route("/",methods = ["GET"])
def home():
    return render_template("predict.html")
@app.route("/predict",methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image)
    
    
    predict_image_generator = ImageDataGenerator(rescale = 1.0/255)
    prediction = model.predict(predict_image_generator.flow(processed_image))
    
    response = {
            "Heart": str(prediction[0][0]),
            "Oblong": str(prediction[0][1]),
            "Oval": str(prediction[0][2]),
            "Round": str(prediction[0][3]),
            "Square": str(prediction[0][4]),
            "Result" : str(max([prediction[0][0],prediction[0][1],prediction[0][2],prediction[0][3],prediction[0][4]]))

            }
    

    return json.dumps(response)

if __name__=='__main__':
    app.run(debug=True)
    
    








    
    

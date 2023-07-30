from flask import Flask, render_template, request
import tensorflow
import os
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
from werkzeug.utils import secure_filename
app = Flask(__name__)

class_names=['air hockey',
 'ampute football',
 'archery',
 'arm wrestling',
 'axe throwing',
 'balance beam',
 'barell racing',
 'baseball',
 'basketball',
 'baton twirling',
 'bike polo',
 'billiards',
 'bmx',
 'bobsled',
 'bowling',
 'boxing',
 'bull riding',
 'bungee jumping',
 'canoe slamon',
 'cheerleading',
 'chuckwagon racing',
 'cricket',
 'croquet',
 'curling',
 'disc golf',
 'fencing',
 'field hockey',
 'figure skating men',
 'figure skating pairs',
 'figure skating women',
 'fly fishing',
 'football',
 'formula 1 racing',
 'frisbee',
 'gaga',
 'giant slalom',
 'golf',
 'hammer throw',
 'hang gliding',
 'harness racing',
 'high jump',
 'hockey',
 'horse jumping',
 'horse racing',
 'horseshoe pitching',
 'hurdles',
 'hydroplane racing',
 'ice climbing',
 'ice yachting',
 'jai alai',
 'javelin',
 'jousting',
 'judo',
 'lacrosse',
 'log rolling',
 'luge',
 'motorcycle racing',
 'mushing',
 'nascar racing',
 'olympic wrestling',
 'parallel bar',
 'pole climbing',
 'pole dancing',
 'pole vault',
 'polo',
 'pommel horse',
 'rings',
 'rock climbing',
 'roller derby',
 'rollerblade racing',
 'rowing',
 'rugby',
 'sailboat racing',
 'shot put',
 'shuffleboard',
 'sidecar racing',
 'ski jumping',
 'sky surfing',
 'skydiving',
 'snow boarding',
 'snowmobile racing',
 'speed skating',
 'steer wrestling',
 'sumo wrestling',
 'surfing',
 'swimming',
 'table tennis',
 'tennis',
 'track bicycle',
 'trapeze',
 'tug of war',
 'ultimate',
 'uneven bars',
 'volleyball',
 'water cycling',
 'water polo',
 'weightlifting',
 'wheelchair basketball',
 'wheelchair racing',
 'wingsuit flying']
class_names=[i.title() for i in class_names]

@app.route('/',methods=['GET'])
def index():
    return render_template('webpage.html')

MODEL=load_model('sportclass_model.h5')

def predict_class(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    pred=MODEL.predict(x)
    predicted_sport=class_names[np.argmax(pred[0])]
    return predicted_sport

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict_class(file_path)
        result=preds
        return result
    return None

    
if __name__ == '__main__':
    app.run(debug=True)
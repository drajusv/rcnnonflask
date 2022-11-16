from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import models
import torch
import jsonpickle
import json
from skimage.color import rgba2rgb
from skimage.io import imread
from PIL import Image, ImageDraw
import os


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)



# Load your trained model
model = models.rcnn()
model.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
model.eval()
basepath = os.path.dirname(__file__)
file_name="box123.jpeg"
app.config["IMAGE_UPLOADS"] = "/Users/drang/Downloads/PyTorch-Object-Detection-Faster-RCNN-Tutorial/pytorch_faster_rcnn_tutorial/project"
app.config['TEMPLATES_AUTO_RELOAD'] = True

print('Model loaded. Check http://127.0.0.1:5000/')
#print(model)
def model_predict(img_path, model):
    im = imread(img_path)
    if im.shape[-1] == 4:
            img = rgba2rgb(im)
    img_out = (im - np.min(im)) / np.ptp(im) #0-1 clip like
    model.eval()
    with torch.no_grad():
        imgy_out = np.transpose(img_out, (2,0,1))
        imgy_out = torch.from_numpy(imgy_out).type(torch.float32)
        pred = model([imgy_out])  
        pred = {key: value.numpy() for key, value in pred[0].items()}
     
    return pred, im

def draw_box(pred, im):
    aallll = []
    L = {}
    c = 0
    #choosing top score boxes
    for i, j in pred.items():

        if i == "scores":
            print(len(j))
            for k in range(len(j)):
                if j[k]>0.5000:

                    L[c]=j[k]

                    c+=1
                else:
                    c+=1
    for i, j in pred.items():
            if i == "boxes":
                print(L.items())
                aallll =  np.array(j[:len(L.keys())])
    print("boundingbox", aallll)
    

    boxes = aallll
    imge = Image.fromarray(im)
    for i in range(len(boxes)):
        ImageDraw.Draw(imge).rectangle(list(boxes[i]), width=5, outline="red")
    
    imge.save(file_name)
    return imge

        

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template("index.html")
    
    


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make prediction
       
        preds, img = model_predict(file_path, model)
        im = draw_box(preds, img)
        im.show()
        print("done")
        aa = os.getcwd()
        full = os.path.join(aa, 'box.jpeg')
        filename = secure_filename("box.jpeg")
        
        return render_template("index.html", filename = "box.jpg")
    return None


@app.route ( "/display" )
def display_image(filename=file_name):
    print(filename)
    return send_file(filename, mimetype='image/jpeg')
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    


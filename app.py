from flask import Flask, render_template, session, redirect, url_for, session,request

import keras.models
import numpy as np  
import scaler
import h5py
import boto3
import os


s3 = boto3.resource('s3',aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],aws_secret_access_key= os.environ['AWS_SECRET_ACCESS_KEY'] )

s3.Object(os.environ['S3_BUCKET_NAME'], 'model_one.h5').download_file( f'model_one.h5') # Python 3.6+


def return_prediction(model,scaler,image):
    #function takes image, converts to numpy array preprocesses it, then predicts what case is

    #we need to create our own scaler for user data preprocess
    imagedata = scaler.scale(image)

    classes = np.array(['Mild Dementia', 'Moderate Dementia','No Dementia','Very Mild Dementia'])
    guess = model.predict_on_batch(imagedata)
    class_ind = np.argmax(guess)
    
    return classes[class_ind],guess

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

model =  keras.models.load_model('model_one.h5')   

@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html')
        
@app.route('/prediction', methods = ['POST','GET'])
def prediction():

    results, percents = return_prediction(model=model,scaler=scaler,image=request.files['file'] )

    vm = np.around((100*percents[0,3]),3)
    milddem = np.around((100*percents[0,1]),3)

    nondem = np.around((100*percents[0,2]),3)
    moddem = np.around((100*percents[0,1]),3)

    mildemw = str(np.max([milddem,1])) + "%"
    moddemw = str(np.max([moddem,1])) + "%"
    nondemw = str(np.max([nondem,1])) + "%"

    vmdw = str(np.max([vm,1])) + "%"
 
    return render_template('prediction.html',results=results,milddem= milddem,moddem=moddem,nondem=nondem,vmd=vm, mildemw=mildemw,moddemw=moddemw,nondemw=nondemw,vmdw=vmdw)

if __name__ == '__main__':
    app.run(debug=True)

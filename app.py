from flask import Flask, render_template, session, redirect, url_for, session,request
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
from keras.models import Model

import numpy as np  
import tensorflow.keras
import joblib
import scaler

def return_prediction(model,scaler,image):
    #function takes image, converts to numpy array preprocesses it, then predicts what case is

    #we need to create our own scaler for user data preprocess
    imagedata = scaler.scale(image)
    

    classes = np.array(['mild-demented', 'moderate-demented','non-demented','very-mild-demented'])
    class_ind = np.argmax(model.predict_on_batch(imagedata))
    
    return classes[class_ind]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

model =  tensorflow.keras.models.load_model('model_one.h5')   
class ALZForm(FlaskForm):
    link_img = TextField('Link to Image')

    submit = SubmitField('Analyze')

@app.route('/', methods=['GET', 'POST'])
def index():

    form = ALZForm()

    if form.validate_on_submit():

        session['link_img'] = form.link_img.data

        return redirect(url_for("prediction"))

    return render_template('index.html', form=form)

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        return render_template("success.html", name = f.filename)  
        
@app.route('/prediction', methods = ['POST','GET'])
def prediction():

 

    results = return_prediction(model=model,scaler=scaler,image=request.files['file'] )

    
    return render_template('prediction.html',results=results)

if __name__ == '__main__':
    app.run(debug=True)

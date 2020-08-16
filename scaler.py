from tensorflow.keras.preprocessing import image
from PIL import Image
from flask import request
import numpy as np

# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def scale(imageurl):
    aimage = load_img(imageurl,target_size=(178,208))
    n = img_to_array(aimage)
    n = [n/255]
    return np.array(n)






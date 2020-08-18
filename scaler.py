from tensorflow.keras.preprocessing import image
from PIL import Image
from flask import request
import numpy as np
from PIL import Image
import cv2
import io

# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import smart_resize

def scale(image):
    
    photo = image
    photo.save('img.jpg')

    a = load_img('img.jpg',target_size=(178,208,3))
    a = img_to_array(a)
    n = [a/255]
    return np.array(n)






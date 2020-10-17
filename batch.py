# First, we import all the necessary dependancies required to perform spam automation
import os
import numpy as np
from glob import glob
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from model.lenet import LeNet
import cv2

path = 'batch/'
path.replace('//', '/') 
path.replace(' ', '\\ ') #Replacing spaces with their escaped versions

#Secondly, we are importing the model that we have trained previously on our dataset
model = load_model('slides.model')

#Creating a new folder called notes to which the slides will be transferred tp
notes_path = 'batch/notes/'
if not os.path.exists(notes_path):
    os.mkdir(notes_path)

def predict(file_path):
    '''
    predict if an image belongs to a slide or not
    '''
    image = cv2.imread(file_path)
    orig = image.copy()
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    (notslides, slides) = model.predict(image)[0]
    label = "Slide" if slides > notslides else "Not a Slide"
    return label


#Extracting the file paths at once using the glob function
files = glob(path + '*.*')

#Looping through the images in the folder and predicting if they are an image of a slide or not

for count, file_path in enumerate(files):
    if predict(file_path)=="Slide": # check if the file belongs to a slide
        file_name = file_path.split('/')[-1] # Getting the orignial file
        os.rename(file_path, notes_path + file_name) #Moving the file to the notes folder
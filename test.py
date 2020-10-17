# First, we import all the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# Secondly we create the command line arguments so as to pass the input through the command line 
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# Thirdly, we load the image from the image path that has been passed by the user
image = cv2.imread(args["image"])
orig = image.copy()

# Now, we resize the image and perform pre-processing inorder to run on our model
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# We now load the CNN that has been trained by us before in the train.py file
print("[UPDATE] The trained model is now being brought in.")
model = load_model(args["model"])

# Using the predict function, we now predict if the image is a slide or not
(notslides, slides) = model.predict(image)[0]

# After getting the required output, we now build the image label based on our input 
label = "Slide" if slides > notslides else "Not a Slide"
proba = slides if slides > notslides else notslides
label = "{}: {:.2f}%".format(label, proba * 100)

# After getting the label, we now display it on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_COMPLEX,
            0.7, (0, 0, 0), 2)

# The final image along with the predicted label is shown after the task
cv2.imshow("Output", output)
cv2.waitKey(0)

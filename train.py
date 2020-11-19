#First, we import all the necessary dependancies that might be useful in training the project
import os
import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from model.vgg16 import VGG16
from model.lenet import LeNet
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")

#Now, we are creating the command line arguments so as to pass the inputs to train the model
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="type the path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="type the path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="type the path to output loss/accuracy plot")
args = vars(ap.parse_args())

#Thirdly, we are initialising hte number of epoches, learning rate and the batch size for training
EPOCHS = 25
INIT_LR = 1e-3
BS = 64

#Now, we are creating an empty list of data and labels to initialise them
print("[UPDATE]: The images are being loaded.")
data = []
labels = []

#Now, we pull in the image path and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

#Now, we loop over all the images that we have been provided
for imagePath in imagePaths:
    #Now, we load the image, pre-process it, and append it to the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)
    #Now, we extract the class label from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "slides" else 0
    labels.append(label)

#Now, we scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

#Now, we perform the train_test data split by allocating 75% for training the data and the remaining 25% for testing the data partition the data
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)

#Now, we convert the datatype of labels to  vectors from originally being integers                                                  
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

#Now, we construct the image generator to perform data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

#After preprocessing the data, now we bring in the model to train it with the data
print("[UPDATE]: We are now compiling the model.")
model = LeNet.build(width=28, height=28, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
#Now that compiling the model is done, we will now train the data
print("[UPDATE]: We are now training the model.")
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(testX, testY), steps_per_epoch=len(
                  trainX) // BS,
              epochs=EPOCHS, verbose=1)

#Now that the training is done, we will save the model to the same directory
print("[UPDATE]: Training is done. We are serialising the network.")
model.save(args["model"], save_format="h5")

# We are now trying to visualise the performance of the model using matplotlib
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Slides/Not Slides")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

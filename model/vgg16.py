#Importing all the required packages to build the VGG16 architecture.
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Activation

#Creating the model class
class VGG16:
    @staticmethod
    def build(depth,height,width,classes):
        inputshape=(depth,width,height)
        #initialising the model
        model=Sequential([
            Conv2D(64,(3,3),input_shape=inputshape,activation='relu',padding='same'),
            Conv2D(64,(3,3),activation='relu',padding='same'),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(128,(3,3),activation='relu',padding='same'),
            Conv2D(128,(3,3),activation='relu',padding='same'),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(256,(3,3),activation='relu',padding='same'),
            Conv2D(256,(3,3),activation='relu',padding='same'),
            Conv2D(256,(3,3),activation='relu',padding='same'),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(512,(3,3),activation='relu',padding='same'),
            Conv2D(512,(3,3),activation='relu',padding='same'),
            Conv2D(512,(3,3),activation='relu',padding='same'),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(512,(3,3),activation='relu',padding='same'),
            Conv2D(512,(3,3),activation='relu',padding='same'),
            Conv2D(512,(3,3),activation='relu',padding='same'),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Flatten(),
            Dense(4096,activation='relu'),
            Dense(4096,activation='relu'),
            Dense(classes,activation='softmax')
        ])
        return model

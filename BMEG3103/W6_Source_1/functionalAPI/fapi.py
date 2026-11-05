#import the necessary libraries
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Dropout

#create the input layer
input_layer = Input(shape=(1, ))
 
#create the other hidden layers and the output layer
hidden_1 = Dense(64, activation='relu')(input_layer)
hidden_2 = Dense(32, activation='relu')(hidden_1)
output = Dense(1, activation='linear')(hidden_2)

model = Model(inputs=input_layer, outputs=output)
model.summary()
from keras.utils import plot_model
 
plot_model(model)
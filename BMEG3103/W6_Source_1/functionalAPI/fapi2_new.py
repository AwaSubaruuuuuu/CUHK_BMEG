# fapi2_new.py - fixed version of fapi2.py
#
# Fix for Keras 3:
#   from keras.layers import merge          -> removed
#   merge.concatenate([h1, h2])             -> keras.layers.concatenate([h1, h2])

# import the necessary libraries
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import concatenate

# create the input layer
input_layer = Input(shape=(1, ))

# create two hidden layers with shared input layer
hidden_1 = Dense(64, activation="relu")(input_layer)
hidden_2 = Dense(64, activation="relu")(input_layer)

# merge the two layers
merged_layers = concatenate([hidden_1, hidden_2])

# build the rest of the layers
hidden_3 = Dense(32, activation="relu")(merged_layers)
output = Dense(1, activation="linear")(hidden_3)

model = Model(inputs=input_layer, outputs=output)
model.summary()

from keras.utils import plot_model
plot_model(model, to_file="fapi2_model.png")
print("Saved model graph to fapi2_model.png")

# fapi_answer1_new.py - fixed version of fapi_answer1.py
#
# Fixes for Keras 3:
#   keras.layers.convolutional.Conv2D        -> keras.layers.Conv2D
#   keras.layers.pooling.MaxPooling2D        -> keras.layers.MaxPooling2D
#   keras.layers.merge.concatenate           -> keras.layers.concatenate
# (Keras 3 collapsed all layer submodules into keras.layers)

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import concatenate

# input layer
visible = Input(shape=(64, 64, 1))
# first feature extractor
conv1 = Conv2D(32, kernel_size=4, activation="relu")(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
flat1 = Flatten()(pool1)
# second feature extractor
conv2 = Conv2D(16, kernel_size=8, activation="relu")(visible)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat2 = Flatten()(pool2)
# merge feature extractors
merge = concatenate([flat1, flat2])
# interpretation layer
hidden1 = Dense(10, activation="relu")(merge)
# prediction output
output = Dense(1, activation="sigmoid")(hidden1)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file="shared_input_layer.png")
print("Saved model graph to shared_input_layer.png")

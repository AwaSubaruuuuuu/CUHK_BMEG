# fapi_answer2_new.py - fixed version of fapi_answer2.py
#
# Fixes for Keras 3:
#   keras.layers.recurrent.LSTM      -> keras.layers.LSTM
#   keras.layers.merge.concatenate   -> keras.layers.concatenate

# Shared Feature Extraction Layer
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import concatenate

# define input
visible = Input(shape=(100, 1))
# feature extraction
extract1 = LSTM(10)(visible)
# first interpretation model
interp1 = Dense(10, activation="relu")(extract1)
# second interpretation model
interp11 = Dense(10, activation="relu")(extract1)
interp12 = Dense(20, activation="relu")(interp11)
interp13 = Dense(10, activation="relu")(interp12)
# merge interpretation
merge = concatenate([interp1, interp13])
# output
output = Dense(1, activation="sigmoid")(merge)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file="shared_feature_extractor.png")
print("Saved model graph to shared_feature_extractor.png")

# tf3.py - Keras 3 / TF 2.21 compatible version of tf2.py
#
# tf2.py used APIs that were removed in Keras 3:
#   - keras.preprocessing.image.ImageDataGenerator  -> keras.utils.image_dataset_from_directory
#   - model.fit_generator()                        -> model.fit()
#   - optimizers.SGD(lr=...)                       -> optimizers.SGD(learning_rate=...)
#   - ModelCheckpoint(period=...)                  -> removed (save every epoch by default)
#   - monitor 'val_acc'                            -> 'val_accuracy'
#   - .h5 full-model saves                         -> .keras format (recommended)

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # silence the oneDNN notice

import keras
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.utils import image_dataset_from_directory

# ---------------------------------------------------------------------------
# Data loading (replaces ImageDataGenerator + flow_from_directory)
# label_mode="categorical" gives one-hot labels, matching categorical_crossentropy
# ---------------------------------------------------------------------------
traindata = image_dataset_from_directory(
    directory="./train",
    image_size=(224, 224),
    label_mode="categorical",
    batch_size=32,
    shuffle=True,
)
testdata = image_dataset_from_directory(
    directory="./test",
    image_size=(224, 224),
    label_mode="categorical",
    batch_size=32,
    shuffle=False,
)

print("Class names:", traindata.class_names)

# VGG16 pretrained on ImageNet
from keras.applications.vgg16 import VGG16

vggmodel = VGG16(weights="imagenet", include_top=True)
vggmodel.summary()

# Freeze all but the last fully-connected layers (first 19 layers frozen, as in tf2.py)
for layers in vggmodel.layers[:19]:
    print(layers)
    layers.trainable = False

# Replace the classifier head with a 3-class output
X = vggmodel.layers[-2].output
predictions = Dense(3, activation="softmax")(X)
model_final = Model(vggmodel.input, predictions)

model_final.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9),
    metrics=["accuracy"],
)

model_final.summary()

# ---------------------------------------------------------------------------
# Training (replaces fit_generator)
# ---------------------------------------------------------------------------
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(
    "vgg16_1.keras",          # new recommended format (was .h5)
    monitor="val_accuracy",   # 'val_acc' no longer exists
    verbose=1,
    save_best_only=True,
)
early = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=40,
    verbose=1,
    mode="auto",
)

model_final.fit(
    traindata,
    steps_per_epoch=2,
    epochs=50,
    validation_data=testdata,
    validation_steps=1,
    callbacks=[checkpoint, early],
)

model_final.save_weights("vgg16_1.weights.h5")

# testimage.py - Run inference on a new image with the trained VGG16 model
#
# Usage:
#   python testimage.py path/to/image.jpg
#   python testimage.py path/to/image.jpg cats dogs elephants   (optional: custom class names)
#
# Loads the trained model saved by tf3.py (vgg16_1.keras, or the weights
# file vgg16_1.weights.h5 as fallback) and predicts which class the
# new image belongs to.

import sys
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # silence the oneDNN notice

import numpy as np
import keras
from keras.models import load_model, Model
from keras.layers import Dense
from keras.utils import load_img, img_to_array

IMAGE_SIZE = (224, 224)
MODEL_PATH = "vgg16_1.keras"
WEIGHTS_PATH = "vgg16_1.weights.h5"


def get_model():
    """Load the trained model - full model first, otherwise rebuild + weights."""
    if os.path.exists(MODEL_PATH):
        print(f"Loading full model from {MODEL_PATH} ...")
        return load_model(MODEL_PATH)

    if os.path.exists(WEIGHTS_PATH):
        # Rebuild the same architecture used in tf3.py, then load the weights
        print(f"{MODEL_PATH} not found, rebuilding architecture and "
              f"loading weights from {WEIGHTS_PATH} ...")
        from keras.applications.vgg16 import VGG16

        vggmodel = VGG16(weights="imagenet", include_top=True)
        for layer in vggmodel.layers[:19]:
            layer.trainable = False
        X = vggmodel.layers[-2].output
        predictions = Dense(3, activation="softmax")(X)
        model = Model(vggmodel.input, predictions)
        model.load_weights(WEIGHTS_PATH)
        return model

    raise FileNotFoundError(
        f"Neither {MODEL_PATH} nor {WEIGHTS_PATH} found. "
        "Run tf3.py first to train and save the model."
    )


def predict_image(model, image_path, class_names):
    # Load and preprocess exactly like the training data:
    # resized to 224x224, raw pixel values (0-255), no extra scaling
    img = load_img(image_path, target_size=IMAGE_SIZE)
    x = img_to_array(img)              # shape: (224, 224, 3), floats 0-255
    x = np.expand_dims(x, axis=0)      # shape: (1, 224, 224, 3) - add batch dim

    preds = model.predict(x, verbose=0)[0]

    print(f"\nImage: {image_path}")
    print("-" * 40)
    for name, p in zip(class_names, preds):
        bar = "#" * int(round(p * 30))
        print(f"{name:<12} {p:6.2%}  {bar}")
    print("-" * 40)
    idx = int(np.argmax(preds))
    print(f"Prediction: {class_names[idx]} "
          f"({preds[idx]:.2%} confidence)")
    return class_names[idx], preds


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testimage.py path/to/image.jpg [class1 class2 class3]")
        sys.exit(1)

    image_path = sys.argv[1]

    # Class names: from command line, or alphabetical folder names in ./train
    # (must match the order Keras used during training)
    if len(sys.argv) > 2:
        class_names = sys.argv[2:]
    elif os.path.isdir("./train"):
        class_names = sorted(os.listdir("./train"))
    else:
        class_names = ["cats", "dogs", "elephants"]

    print("Classes:", class_names)

    model = get_model()
    predict_image(model, image_path, class_names)

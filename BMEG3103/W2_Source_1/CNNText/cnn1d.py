import keras
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  #CPU only

batch_size = 32
epochs = 3
maxlen = 10000
batch_size = 32
embedding_dim = 128
num_filters = 64
kernel_size = 5

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

word_index = reuters.get_word_index(path="reuters_word_index.json")
num_classes = max(y_train) + 1 
index_to_word = {}
for key, value in word_index.items():
    index_to_word[value] = key
    
    
tokenizer = Tokenizer(num_words=maxlen)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')



y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(layers.Embedding(512, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())


history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


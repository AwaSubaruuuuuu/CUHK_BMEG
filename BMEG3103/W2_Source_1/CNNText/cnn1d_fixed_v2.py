import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras import layers
import numpy as np
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # CPU only

batch_size = 32
epochs = 3
maxlen = 10000
embedding_dim = 128
num_filters = 64
kernel_size = 5

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

word_index = reuters.get_word_index(path="reuters_word_index.json")
num_classes = max(y_train) + 1
index_to_word = {}
for key, value in word_index.items():
    index_to_word[value] = key

# 修改点 1：Keras 3 移除了 keras.preprocessing.text.Tokenizer，
# 这里用等价的 numpy 实现替换 tokenizer.sequences_to_matrix(mode='binary')：
# 生成 (样本数, maxlen) 的 0/1 矩阵，某词在样本中出现过则对应位置置 1
def sequences_to_binary_matrix(sequences, num_words):
    matrix = np.zeros((len(sequences), num_words), dtype="int32")
    for i, seq in enumerate(sequences):
        for word_id in seq:
            if word_id < num_words:
                matrix[i, word_id] = 1
    return matrix

x_train = sequences_to_binary_matrix(x_train, maxlen)
x_test = sequences_to_binary_matrix(x_test, maxlen)

# 修改点 2：Keras 3 里 np_utils 已移除，直接用 keras.utils.to_categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# 修改点 3：Keras 3 的 Embedding 不再接受 input_length 参数，改用 Input 层声明输入形状
model.add(keras.Input(shape=(maxlen,)))
model.add(layers.Embedding(512, embedding_dim))
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

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

text = "hello world! this is a simple text generation example using rnn. " * 200
chars = sorted(list(set(text)))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

vocab_size = len(chars)  #30
seq_length = 20  #read 20 characters and predict the next one

input_sequences, target_chars = [], []
for i in range(len(text) - seq_length):
    input_sequences.append(text[i:i + seq_length])
    target_chars.append(text[i + seq_length])
 
#Explanation 
#input_sequences = ["hel", "ell"]
#target_chars    = ['l',   'o']
#Interpretation:

#Input (context)	Target (next char)
#"hel"	'l'
#"ell"	'o'    
    

X = np.zeros((len(input_sequences), seq_length), dtype=int)
y = np.zeros((len(input_sequences), vocab_size), dtype=int)
for i, seq in enumerate(input_sequences):
    for t, char in enumerate(seq):
        X[i, t] = char_to_index[char]
    y[i, char_to_index[target_chars[i]]] = 1
    
    
#Verify against the vocabulary {'e':0, 'h':1, 'l':2, 'o':3}:
#X[0] = [1, 0, 2] ➡️ [h, e, l] = "hel" ✅
#X[1] = [0, 2, 2] ➡️ [e, l, l] = "ell" ✅
#y[0] = [0, 0, 1, 0] ➡️ index 2 = 'l' ✅
#y[1] = [0, 0, 0, 1] ➡️ index 3 = 'o' ✅


#X = [[1,0,2], [0,2,2], ...]     ➡️ feed into LSTM
#y = [[0,0,1,0],[0,0,0,1],...]   ➡️ check the answer

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32),   # input_length removed
    LSTM(64),
    Dense(vocab_size, activation='softmax'),
])

#dimension0  dimension1  dimension2  ...  dimension31
#index 0: [ 0.2  -0.5   0.8   ...   0.1 ]   ➡️ 'a' vector
#index 1: [-0.3   0.9  -0.1   ...   0.4 ]   ➡️ 'b' vector
#index 2: [ 0.7   0.1   0.3   ...  -0.2 ]   ➡️ 'c' vector
#...
#index 29:[ ...                         ]   ➡️ last character


# of parameters
#= 4 脳 [(32 + 64) 脳 64 + 64]
#= 4 脳 [6144 + 64]
#= 24,832

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=200, verbose=2, batch_size=64)


def generate_text(seed_text, length=50, temperature=0.5):
    result = seed_text
    for _ in range(length):
        encoded = [char_to_index.get(c, 0) for c in result[-seq_length:]]
        padded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        preds = model.predict(padded, verbose=0)[0].astype("float64")

        preds = np.log(preds + 1e-9) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))

        next_index = np.random.choice(len(preds), p=preds)
        result += index_to_char[next_index]
    return result


print(generate_text("hello world", length=60, temperature=0.5))

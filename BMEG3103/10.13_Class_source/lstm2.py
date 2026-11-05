import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ----------------------------------------------------------------------
# 1. 30 training sentences:15 + + 15 -
# ----------------------------------------------------------------------
positive = [
    "i love this movie",
    "this film is great",
    "the story was amazing",
    "what a wonderful show",
    "the acting is excellent",
    "i really enjoyed it",
    "this is my favorite film",
    "the plot was fantastic",
    "beautiful and touching story",
    "the ending made me happy",
    "great performance by everyone",
    "i would watch it again",
    "the movie was so good",
    "perfect from start to finish",
    "i had a wonderful time",
]

negative = [
    "i hate this movie",
    "this film is terrible",
    "the story was boring",
    "what an awful show",
    "the acting is bad",
    "i really disliked it",
    "this is my least favorite film",
    "the plot was confusing",
    "ugly and pointless story",
    "the ending made me angry",
    "poor performance by everyone",
    "i would never watch it again",
    "the movie was so bad",
    "a waste of time from start to finish",
    "i had a terrible time",
]

texts  = positive + negative
labels = [1] * len(positive) + [0] * len(negative)   # 1=positive, 0=negative

print("Total sentences:", len(texts))
for t, l in zip(texts, labels):
    print(f"  [{l}] {t}")

# ----------------------------------------------------------------------
# 2. Tokenize + Padding
# ----------------------------------------------------------------------
vocab_size = 100
maxlen = 8

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=maxlen, padding="post", truncating="post")

y = np.array(labels)

print("\nX shape:", X.shape)
print("Vocabulary:", tokenizer.word_index)

# ----------------------------------------------------------------------
# 3. LSTM model
# ----------------------------------------------------------------------
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=8, input_length=maxlen),
    LSTM(16),
    Dense(1, activation="sigmoid"),
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# ----------------------------------------------------------------------
# 4. train
# ----------------------------------------------------------------------
model.fit(X, y, epochs=30, batch_size=4)
model.summary()


#model.fit(X, y, epochs=30, batch_size=4, verbose=1)

# ----------------------------------------------------------------------
# 5. predict new sentence
# ----------------------------------------------------------------------
def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    score = model.predict(seq, verbose=0)[0][0]
    return score, ("Positive" if score > 0.5 else "Negative")

tests = [
    "i love this film",
    "this movie is terrible",
    "what a great story",
    "boring and awful",
]

print("\nPredictions:")
for t in tests:
    score, label = predict(t)
    print(f"  {t:30s} -> {label} ({score:.3f})")

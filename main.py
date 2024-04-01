import numpy as np
from src.preprocess import load_data
from src.model import build_model, train_model
from src.evaluate import evaluate_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess data
data_path = 'data/spam.csv'
df = load_data(data_path)

# Split data into features and labels
X = df['text'].values
y = df['label'].values

# Convert text to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
vocab_size = len(tokenizer.word_index) + 1

# Pad sequences
max_length = max([len(seq) for seq in X])
X = pad_sequences(X, maxlen=max_length, padding='post')

# Convert labels to binary
y = np.where(y == 'spam', 1, 0)

# Split data into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train model
embedding_dim = 128
model = build_model(vocab_size, embedding_dim, max_length)
model = train_model(model, X_train, y_train, X_val, y_val, epochs=10)

# Evaluate model on test set
X_test, y_test = X_val, y_val  # Using validation set as test set for this example
evaluate_model(model, X_test, y_test)

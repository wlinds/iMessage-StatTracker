import os
import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense
from keras.optimizers import Adam

from utils.csv_cleaner import equalize_dataset

class EmotionClassifier:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.input_dim = None
        self.input_length = None
        self.dense_output = None
        self.one_hot_labels = None

    def encode_labels(self, df):
        integer_labels = self.label_encoder.fit_transform(df['label'])
        self.one_hot_labels = to_categorical(integer_labels)
        return self.one_hot_labels

    def get_dims(self, df):
        self.tokenizer.fit_on_texts(df['text'])
        self.input_dim = len(self.tokenizer.word_index) + 1
        self.input_length = max(len(sequence) for sequence in self.tokenizer.texts_to_sequences(df['text']))
        self.dense_output = len(df['label'].value_counts())

    def get_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.input_dim, output_dim=128, input_length=self.input_length))
        model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(self.dense_output, activation='softmax'))
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_data(self, df):
        sequences = self.tokenizer.texts_to_sequences(df['text'])
        X = pad_sequences(sequences, maxlen=self.input_length, padding='post', truncating='post')
        X_train, X_test, y_train, y_test = train_test_split(X, self.one_hot_labels, test_size=0.01, random_state=42)
        return X_train, X_test, y_train, y_test

    def fit_model(self, epochs=10, batch_size=32, save_dir="pretrained", model_prefix="alpha"):
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test))

        if save_dir:
            pretrained_dir = os.path.join(os.getcwd(), save_dir, model_prefix)
            os.makedirs(pretrained_dir, exist_ok=True)

            # time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            time = 0
            name = f'{model_prefix}_{time}'
            save = os.path.join(pretrained_dir, name + ".keras")

            self.model.save(save)
            print(f"Model saved to {save}")

            label_encoder_save_path = os.path.join(pretrained_dir, f'{model_prefix}_label_encoder.pkl')
            joblib.dump(self.label_encoder, label_encoder_save_path)
            print(f"Label encoder saved to {label_encoder_save_path}")

            tokenizer_save_path = os.path.join(pretrained_dir, f'{model_prefix}_tokenizer.pkl')
            joblib.dump(self.tokenizer, tokenizer_save_path)
            print(f"Tokenizer saved to {tokenizer_save_path}")

            dims_save_path = os.path.join(pretrained_dir, f'{model_prefix}_dims.pkl')
            joblib.dump((self.input_dim, self.input_length), dims_save_path)
            print(f"Input dimensions saved to {dims_save_path}")

        return history

    def load_model_and_label_encoder(self, model_path, label_encoder_path):
        self.model = load_model(model_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def load_tokenizer_and_dims(self, tokenizer_path, dims_path):
        self.tokenizer = joblib.load(tokenizer_path)
        self.input_dim, self.input_length = joblib.load(dims_path)

    def predict_label(self, input_text):
        if self.model is None or self.label_encoder is None or self.tokenizer is None:
            raise ValueError("Model, label encoder, or tokenizer not loaded. Call load_model_and_label_encoder method first.")

        sequence = self.tokenizer.texts_to_sequences([input_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.input_length, padding='post', truncating='post')

        predicted_probabilities = self.model.predict(padded_sequence)
        predicted_label_index = np.argmax(predicted_probabilities, axis=1)

        predicted_label = self.label_encoder.inverse_transform(predicted_label_index)

        return predicted_label[0]

    def train(self, df, epochs=3):
        self.one_hot_labels = self.encode_labels(df)
        self.get_dims(df)
        self.model = self.get_model()
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_data(df)
        self.fit_model(epochs=epochs)

if __name__ == "__main__":
    train = 0
    classifier = EmotionClassifier()

    if train:
        df = pd.read_csv('data/emotions-sv_fix.csv')
        df = df[df['label'] != 'surprise']

        df = equalize_dataset(df, min_samples=2048)
        classifier.train(df, epochs=25)

    model_path = "pretrained/alpha/alpha_0.keras"
    label_encoder_path = "pretrained/alpha/alpha_label_encoder.pkl"
    tokenizer_path = "pretrained/alpha/alpha_tokenizer.pkl"
    dims_path = "pretrained/alpha/alpha_dims.pkl"

    classifier.load_model_and_label_encoder(model_path, label_encoder_path)
    classifier.load_tokenizer_and_dims(tokenizer_path, dims_path)

    input_text = "han blev ledsen för att riset inte va gott"
    predicted_label = classifier.predict_label(input_text)
    print(f"Predicted Label: {predicted_label}")

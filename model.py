import os
import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense
from keras.optimizers import Adam

from tensorflow.keras.models import load_model

tokenizer = Tokenizer()
label_encoder = LabelEncoder()

def encode_labels(df):
    integer_labels = label_encoder.fit_transform(df['label'])
    one_hot_labels = to_categorical(integer_labels)
    
    return one_hot_labels


def get_dims(df):
    tokenizer.fit_on_texts(df['text'])
    input_dim = len(tokenizer.word_index) + 1
    input_length = max(len(sequence) for sequence in tokenizer.texts_to_sequences(df['text']))

    return input_dim, input_length


def get_model(input_dim, input_length):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=input_length))
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_data(df):
    sequences = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(sequences, maxlen=input_length, padding='post', truncating='post')
    X_train, X_test, y_train, y_test = train_test_split(X, one_hot_labels, test_size=0.01, random_state=42)

    return X_train, X_test, y_train, y_test


# TODO Consider adding early stopping and other regularizations

def fit_model(model, epochs=10, batch_size=32, save_dir="pretrained", model_prefix="alpha"):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    if save_dir:
        pretrained_dir = os.path.join(os.getcwd(), save_dir, model_prefix)
        os.makedirs(pretrained_dir, exist_ok=True)

        time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'{model_prefix}_{time}'
        save = os.path.join(pretrained_dir, name + ".keras")

        model.save(save)
        print(f"Model saved to {save}")

        label_save_path = os.path.join(pretrained_dir, f'{model_prefix}_one_hot_labels.pkl')

        joblib.dump(y_train, label_save_path)
        print(f"One-hot encoded labels saved to {label_save_path}")

    return history



def predict_label(model_path, tokenizer, input_text, input_length, label_path):
    loaded_model = load_model(model_path)

    one_hot_labels = joblib.load(label_path)

    sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=input_length, padding='post', truncating='post')

    # This is a mess, should be refactored

    predicted_probabilities = loaded_model.predict(padded_sequence)

    predicted_label_index = np.argmax(predicted_probabilities, axis=1)

    predicted_label = one_hot_labels[predicted_label_index]

    return predicted_label[0]



if __name__ == "__main__":
    df = pd.read_csv('data/emotions-sv_fix.csv')

    one_hot_labels = encode_labels(df)

    input_dim, input_length = get_dims(df)

    X_train, X_test, y_train, y_test = get_data(df)

    model = get_model(input_dim, input_length)

    # fit_model(model, epochs=3)

    model_path = "pretrained/alpha/alpha_20240116_175607.keras"
    label_path = "pretrained/alpha/alpha_one_hot_labels.pkl"

    predicted_label = predict_label(model_path, tokenizer, "Your input text here.", input_length, label_path)
    print(f"Predicted Label: {predicted_label}")

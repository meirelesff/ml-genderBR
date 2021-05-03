from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras
from config import settings
import pipelines as pps
import pandas as pd
import numpy as np
import model # Local module


# Function to make index-characters
def make_char_index(nomes):
    chars = set("".join(nomes))
    chars.add("DUMMY")
    return dict((c, i) for i, c in enumerate(chars))


# Function to one-hot encode names
def one_hot(nome, pad, chars_index):

    nome_trunc = nome[:pad]
    res = []
    for i in [chars_index[ch] for ch in nome_trunc]:
        nome_vector = np.zeros(len(chars_index))
        nome_vector[i] = 1
        res.append(nome_vector)
    
    for i in range(0, pad - len(res)):
        dummy = np.zeros(len(chars_index))
        dummy[chars_index["DUMMY"]] = 1
        res.append(dummy)

    return np.vstack(res)


# LSTM bidirect model class
class LSTMbidirect(keras.Model):
    def __init__(self, pad, char_len, units = 128, dropout = 0.2):
        super(LSTMbidirect, self).__init__()

        self._lstm = layers.Bidirectional(layers.LSTM(units, return_sequences=True), 
                                          input_shape=(pad, char_len))
        self._dropout = layers.Dropout(dropout)
        self._dense = layers.Dense(1, activity_regularizer=keras.regularizers.l2(0.002))
        self._activation = layers.Activation("sigmoid")

    def call(self, inputs):
        output = self._lstm(inputs)
        output = self._dropout(output)
        output = self._dense(output)
        output = self._activation(output)
        return output
    


if __name__ == "__main__":
  
    # Sample data
    y, nomes = model.prepare_sample("raw_data/nomes.csv", settings)

     # Create char index
    chars_index = make_char_index(nomes)

    # One hot encode names
    x_onehot = list(map(lambda x: one_hot(x, settings.pad, chars_index), nomes))
    x_onehot = np.asarray(x_onehot)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(x_onehot, y, test_size = 0.1,
                                                        random_state=settings.seed)

    # Train model with 128 units
    stop = EarlyStopping(patience=5)
    cpoint = ModelCheckpoint("src/models/best_lstm_128units.h5", save_best_only=True)
    res_lstm = LSTMbidirect(20, len(chars_index), dropout=0.3)
    res_lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = res_lstm.fit(X_train, y_train, batch_size=settings.batch_size, epochs=settings.epochs, 
                           validation_data=(X_test, y_test), 
                           callbacks = [stop, cpoint])

    pd.DataFrame(history.history).to_csv("results/lstm_128_units.csv")

    # Train model with 256 units
    stop = EarlyStopping(patience=5)
    cpoint = ModelCheckpoint("src/models/best_lstm_256units.h5", save_best_only=True)
    res_lstm = LSTMbidirect(20, len(chars_index), units = 256, dropout=0.3)
    res_lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = res_lstm.fit(X_train, y_train, batch_size=settings.batch_size, epochs=settings.epochs, 
                           validation_data=(X_test, y_test), 
                           callbacks = [stop, cpoint])

    pd.DataFrame(history.history).to_csv("results/lstm_256_units.csv")




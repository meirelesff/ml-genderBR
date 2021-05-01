from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow import keras
import model # Local module


# Function to make index-characters
def make_char_index(nomes):
    chars = set("".join(nomes))
    chars.add("DUMMY")
    return dict((c, i) for i, c in enumerate(chars))

# Funcion 






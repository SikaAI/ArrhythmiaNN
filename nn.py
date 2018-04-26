import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_filename):
  array = np.loadtxt(open(csv_filename, "rb"), delimiter=",")
  train_samples = array[:,:-1]
  train_labels  = array[:,279]-1 # subtract 1 to shift classes to 0-15
  train_labels[train_labels > 0] = 1 # convert all classes above 0 to 1 for binary classification

  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_train_samples = scaler.fit_transform((train_samples))

  return (train_labels, scaled_train_samples)

def train_data(train_labels, train_samples):
  model = Sequential([
    Dense(16, input_shape=(279,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
  ])

  model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(train_samples, train_labels, batch_size=10, epochs=100, shuffle=True, verbose=2)


if __name__ == "__main__":
  data = load_data("data.csv")
  train_data(data[0], data[1])

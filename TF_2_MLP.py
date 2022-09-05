"""

This snippet of code is for the set of functions we will use for the basic Multi-Layer Perceptron architecture to try
on the different data.

Note this is built on TensorFlow 2 code and may not work on earlier versions.

In this snippet the architecture is built on the sequential "build" of TF.

"""

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2,l1,l1_l2
from tensorflow import keras

class build_MLP(object):

    def __init__(self, n_steps, n_features, n_labels, type="simple"):

        self.type = type
        self.n_labels = n_labels
        self.n_features = n_features
        self.n_steps = n_steps

        if type == "simple":
            print("Building Simple MLP")
            clf = self.simple()
        elif type == "deep":
            print("Building Stacked MLP")
            clf = self.deep()
        elif type == "simple2":
            print("Building my MLP")
            clf = self.simple2()
        else:
            print("Building Simple MLP")
            clf = self.simple()

        self.model = clf
        return

    def simple(self, n_units=100, dropout=0.1):

        model = Sequential()

        # Add Dense layer with 100 units
        model.add(Dense(n_units,  input_shape=(self.n_features,), activation="relu"))
        model.add(Dropout(dropout))
        model.add(Dense(n_units, kernel_regularizer=l2(0.1), activity_regularizer=l1(0.01),  activation="relu"))

        model.add(Dropout(dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss="categorical_crossentropy", metrics=['accuracy'],optimizer=keras.optimizers.Adam(learning_rate=0.0001))


        print(model.summary())
        return model

    def deep(self, n_units=100, dropout=0.1, n_layers=5):

        model = Sequential()

        # Add Dense layer with 100 units
        model.add(Dense(n_units, input_shape=(self.n_features,), activation="relu"))
        model.add(Dropout(dropout))
        for n in range(n_layers):
            model.add(Dense(n_units, activation="relu"))
        model.add(Dropout(dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

        print(model.summary())
        return model

    def simple2(self, n_units=100, dropout=0.1):

        model = Sequential()

        # Add Flatten layer with 100 units
        model.add(keras.layers.Flatten(input_shape=(self.n_features,)))
        # model.add(Dropout(dropout))
        model.add(Dense(n_units, kernel_regularizer=l2(0.1), activity_regularizer=l1(0.01), activation="relu"))
        model.add(Dropout(dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=keras.optimizers.Adam(learning_rate=0.0001))

        print(model.summary())
        return model

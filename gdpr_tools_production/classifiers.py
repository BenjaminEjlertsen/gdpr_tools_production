from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop

class DNN():

    def __init__(self, input_shape):

        self.model = Sequential([
            Dense(256, input_shape=input_shape),
            Dropout(0.1),
            Activation('relu'),
            Dense(256),
            Dropout(0.25),
            Activation('relu'),
            Dense(512),
            Dropout(0.5),
            Activation('relu'),
            Dense(1),
            Activation('sigmoid'),
        ])

        optimizer = RMSprop(lr=0.001)

        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
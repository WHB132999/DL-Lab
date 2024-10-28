import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Bidirectional, GRU, SimpleRNN, Dropout
from keras.models import Sequential
import gin


# n_hiddens = 32
#
# timesteps = len(X_train[0])
# input_dims = len(X_train[0][0])
#
# n_classes = count_func(Y_train)
@gin.configurable
class ModelRNN(object):
    def __init__(self, dropout_rate, rnn_type, n_lstm, n_dense, dense_units):
        self.dropout_rate = dropout_rate
        self.rnn_type = rnn_type
        self.n_lstm = n_lstm
        self.n_dense = n_dense
        self.dense_units = dense_units

    def my_LSTM(self, n_classes=6, n_hiddens=250):
        model = Sequential(name='my_LSTM')
        model.add(LSTM(n_hiddens))
        model.add(Dropout(rate=self.dropout_rate))
        model.add(Dense(n_classes, activation='softmax'))

        return model


    def my_Bidirectional(self, timesteps, input_dims, n_classes):
        model = Sequential(name='my_Bidirectional')
        model.add(Bidirectional(LSTM(n_hiddens, input_shape=(timesteps, input_dims))))
        model.add(Dropout(rate=0.5, training=True))
        model.add(Dense(n_classes, activation='softmax'))

        return model


    def my_EnsembleLSTM(self, timesteps, input_dims, n_classes):
        model = Sequential(name='my_EnsembleLSTM')
        model.add(LSTM(n_hiddens, return_sequences=True, input_shape=(timesteps, input_dims)))  # Or use bidirectional layer
        model.add(LSTM(n_hiddens))
        model.add(Dropout(rate=0.5, training=True))
        model.add(Dense(n_classes, activation='softmax'))


    def my_GRU(self, timesteps, input_dims, n_classes):
        model = Sequential(name='my_GRU')
        model.add(GRU(n_hiddens, return_sequences=True))
        model.add(GRU(n_hiddens))
        model.add(Dropout(rate=0.5, training=True))
        model.add(Dense(n_classes, activation='softmax'))


    def multi_rnn(self, rnn_units=250):
        model = Sequential()
        for i in range(1, self.n_lstm):
            if self.rnn_type == 'LSTM':
                model.add(LSTM(rnn_units, return_sequences=True))
            elif self.rnn_type == 'simple_rnn':
                model.add(SimpleRNN(rnn_units, return_sequences=True))
            elif self.rnn_type == 'GRU':
                model.add(GRU(rnn_units, return_sequences=True))
            else:
                return ValueError

        if self.rnn_type == 'LSTM':
            model.add(LSTM(rnn_units, return_sequences=False))
        elif self.rnn_type == 'simple_rnn':
            model.add(SimpleRNN(rnn_units, return_sequences=False))
        elif self.rnn_type == 'GRU':
            model.add(GRU(rnn_units, return_sequences=False))
        else:
            return ValueError

        model.add(Dropout(self.dropout_rate))
        for i in range(1, self.n_dense + 1):
            model.add(Dense(self.dense_units / i, activation='relu'))
        model.add(Dense(6, activation='softmax'))

        return model

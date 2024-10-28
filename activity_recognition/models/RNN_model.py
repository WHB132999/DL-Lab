import gin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU


def multi_rnn(rnn_type, n_lstm, dropout_rate, window_size, rnn_units, n_dense=2, dense_units=256):
    model = Sequential()
    for i in range(1, n_lstm):
        if rnn_type == 'LSTM':
            model.add(LSTM(rnn_units, input_shape=(window_size, 6), return_sequences=True))
        elif rnn_type == 'simple_rnn':
            model.add(SimpleRNN(rnn_units, input_shape=(window_size, 6), return_sequences=True))
        elif rnn_type == 'GRU':
            model.add(GRU(rnn_units, input_shape=(window_size, 6), return_sequences=True))
        else:
            return ValueError

    model.add(Dropout(dropout_rate))
    for i in range(1, n_dense+1):
        model.add(Dense(dense_units/i, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    return model



from tensorflow.keras import Sequential, Dense, Flatten, Dropout, LSTM, TimeDistributed, Conv1D, MaxPooling1D


def cnn_lstm_model():
    model = Sequential()
    model.add(TimeDistributed(
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        input_shape=(None, 250, 6)))
    model.add(TimeDistributed(
        Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    return model

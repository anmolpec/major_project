import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, LSTM, Dense


class CNN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer1 = Conv1D(32, 3, activation = 'relu', input_shape = (309, 300), padding = 'same')
        self.layer2 = MaxPool1D(pool_size = 2, strides = 2)
        self.layer3 = Conv1D(16, 3, activation = 'relu', input_shape = (None, 32), padding = 'same')
        self.layer4 = MaxPool1D(pool_size = 2, strides = 2)
    def call(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))



class RNN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer = LSTM(8)
            
    def call(self, x):
        return self.layer(x)



class Combined(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.CNN = CNN()
        self.RNN = RNN()
        self.dense = Dense(1, activation = 'sigmoid') 
    
    def call(self, x):
        return self.dense(self.RNN(self.CNN(x)))




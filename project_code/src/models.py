import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D, MaxPool1D, LSTM, Dense, MultiHeadAttention, LayerNormalization, Flatten


class ResNetToTransformer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resnet = ResNet1D()
        self.transformer_layer_1 = TransformerMini()
        self.transformer_layer_2 = TransformerMini()
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation = 'relu')
        self.fc2 = Dense(1, activation = 'sigmoid')
        
        l = []

        for i in range(154):
            tl = []
            for j in range(128):
                tl.append(np.sin((i + 1) / np.power(10000, 2*j / 256)))
                tl.append(np.cos((i + 1) / np.power(10000, 2*j / 256)))

            l.append(tl)
        
        self.pos_enc = tf.constant(np.array(l), dtype = tf.float32)

    

    def call(self, x):
        out1 = self.transformer_layer_2(self.transformer_layer_1(self.resnet(x) + self.pos_enc))
        out2 = self.flatten(out1)
        return self.fc2(self.fc1(out2))


class TransformerMini(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transformer_layer_1 = TransformerLayer(256, 256)
        self.transformer_layer_2 = TransformerLayer(256, 256)

    def call(self, x):
        return self.transformer_layer_2(self.transformer_layer_1(x))



class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, key_dim, value_dim, **kwargs):
        super().__init__(**kwargs)
        self.MHA = MultiHeadAttention(num_heads = 8, key_dim = key_dim, value_dim = value_dim)
        self.layer_norm_1 = LayerNormalization()
        self.w1 = tf.Variable(np.random.rand(256, 1024), dtype = tf.float32)
        self.b1 = tf.Variable(np.random.rand(1, 1024), dtype = tf.float32)
        self.relu = tf.nn.relu
        self.w2 = tf.Variable(np.random.rand(1024, 256), dtype = tf.float32)
        self.b2 = tf.Variable(np.random.rand(1, 256), dtype = tf.float32)
        self.layer_norm_2 = LayerNormalization()

    def call(self, x):
        mha_out = self.MHA(x, x, x)
        norm_out = self.layer_norm_1(x + mha_out)
        relu_out = self.relu(norm_out@self.w1 + self.b1)
        ff_out = relu_out@self.w2 + self.b2
        norm_out2 = self.layer_norm_2(norm_out + ff_out)
        return norm_out2




class ResNet1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_pre = ResNetPreLayer(256, 64)
        self.conv1 = ResNetLayer1D(64, 64)
        self.conv2 = ResNetLayer1D(64, 128)
        self.conv3 = ResNetLayer1D(128, 128)
        self.conv4 = ResNetLayer1D(128, 256)
        self.conv5 = ResNetLayer1D(256, 256)
        self.conv6 = ResNetLayer1D(256, 256)

    def call(self, x):
        return self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(self.conv_pre(x)))))))


class ResNetPreLayer(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv1D(out_dim, 5, activation = 'relu', input_shape = (None, in_dim), padding = 'same')
        self.max_pool = MaxPool1D(pool_size = 2, strides = 2)
        self.layer_norm = LayerNormalization()
    
    def call(self, x):
        return self.layer_norm(self.max_pool(self.conv(x)))


class ResNetLayer1D(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.layer1 = Conv1D(in_dim, 3, input_shape = (None, in_dim), padding = 'same')
        self.layer_norm_1 = LayerNormalization()
        self.relu = tf.nn.relu
        self.layer2 = Conv1D(out_dim, 3, input_shape = (None, in_dim), padding  = 'same')
        self.layer_norm_2 = LayerNormalization()
        if in_dim == out_dim:
            self.block = tf.constant(np.eye(in_dim, out_dim), dtype = tf.float32)
        else:
            self.block = tf.Variable(np.random.rand(in_dim, out_dim), dtype = tf.float32)

    def call(self, x):
        out1 = self.layer1(x)
        out2 = self.relu(self.layer_norm_1(out1))
        out3 = self.layer2(out2)
        return self.layer_norm_2(out3 + x @ self.block)
    


        

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





import tensorflow as tf

from models import Combined
from encoding import generate_embedding, data_df
from tqdm import tqdm

model = Combined()

bce = tf.keras.losses.BinaryCrossentropy()

def train(model, x, y, learning_rate):
    with tf.GradientTape() as tape:
        loss = bce(y, model(x))
    
    deriv = tape.gradient(loss, model.trainable_variables)

    for v, dv in zip(model.trainable_variables, deriv):
        v.assign_sub(learning_rate*dv)

for i in tqdm(range(1000)):
    train(model, tf.reshape(tf.constant(generate_embedding(i)), (1, 309, 300)), tf.reshape(tf.constant(data_df['classification'][i]), (1, 1)), 0.001)

tf.saved_model.save(model, "model/model_test")




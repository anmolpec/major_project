import tensorflow as tf

from encoding import generate_embedding, data_df


def generate_dataset(base_index, size, batch_size):
    
    assert base_index + size - 1 <= 61000

    features = []
    targets = []
    for i in range(size):
        y = data_df['classification'][base_index + i]
        x = generate_embedding(base_index + i)
        features.append(x)
        targets.append(y)
    
    features = tf.constant(features)
    targets = tf.constant(targets)

    return tf.data.Dataset.from_tensors((features, targets))




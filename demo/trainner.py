import tensorflow as tf


def add_layer(inputs, in_size, out_size, activation_function=None):
    weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    weight_plus_b = tf.matmul(inputs, weight) + biases
    outputs = weight_plus_b if activation_function is None else activation_function(weight_plus_b)
    return outputs

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def add_layer(input, in_size, out_size, n_layer, activation_func=None):
    layer_name = str(n_layer)
    with tf.name_scope('AddLayer'):
        with tf.name_scope('Weight'):
            Weight = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name + '/weight', Weight)
        with tf.name_scope('Biases'):
            ## recommend value is nozero.
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name + 'biases', biases)
        with tf.name_scope('wx_plus_biases'):
            Wx_plus_biases = tf.matmul(input, Weight) + biases
        if activation_func == None:
            outputs = Wx_plus_biases
        else:
            outputs = activation_func(Wx_plus_biases)
        tf.summary.histogram('output',outputs)
        return outputs

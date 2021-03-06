import tensorflow as tf
import numpy as np

from feature_formater import get_train_data

# this is data
lr = 0.001
# training_iters = 100000
batch_size = 100

n_inputs = 49  # MNIST data input (img shape: 28*28)
n_steps = 4  # time steps
n_hidden_unis = 128  # neurons in hidden layer
n_classes = 2  # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weight
weight = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
    'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unis, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weight, biases):
    # hidden layer for input to cell
    ##########################################
    # X(128batch , 28 steps, 28 inputs
    # ===>(128 * 28,28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weight['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_unis])

    # cell
    ##########################################
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
    _init_states = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, status = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_states, time_major=False)

    # hidden layer for output as the final results
    ##########################################

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    result = tf.matmul(outputs[-1], weight['out']) + biases['out']

    return result


pred = RNN(x, weight, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batch_xs, batch_ys = get_train_data()

    step = 0
    index_from = batch_size * step
    index_to = batch_size * (step + 1)

    while index_to < batch_ys.__len__():
        xs = batch_xs[index_from:index_to]
        ys = batch_ys[index_from:index_to]

        sess.run([train_op], feed_dict={
            x: xs,
            y: ys
        })
        if step % 10 == 0:
            print (sess.run([cost, accuracy], feed_dict={
                x: xs,
                y: ys
            }))
        step += 1
        index_from = batch_size * step
        index_to = batch_size * (step + 1)


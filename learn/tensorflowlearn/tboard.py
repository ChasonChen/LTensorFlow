import matplotlib.pyplot as plt
import numpy as np
import addLayer
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = np.linspace(-1, 1, 300, dtype=np.float64)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x.shape)
y = np.square(x) - 0.5 + noise

with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 1], name='xInput')
    ys = tf.placeholder(tf.float32, [None, 1], name='yInput')

l1 = addLayer.add_layer(xs, 1, 10, 1, activation_func=tf.nn.relu)
prediction = addLayer.add_layer(l1, 10, 1, 2, activation_func=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('step'):
    step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
merge = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(step, feed_dict={xs: x, ys: y})
        if i % 20 == 0:
            result = sess.run(merge, feed_dict={xs: x, ys: y})
            writer.add_summary(result, i)

import matplotlib.pyplot as plt
import numpy as np
import addLayer
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = np.linspace(-1, 1, 300, dtype=np.float64)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x.shape)
y = np.square(x) - 0.5 + noise

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.scatter(x, y)
plt.ion()
plt.show()

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = addLayer.add_layer(xs, 1, 10, activation_func=tf.nn.relu)
prediction = addLayer.add_layer(l1, 10, 1, activation_func=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(step, feed_dict={xs: x, ys: y})
        if i % 50 == 0:
            loss_value = sess.run(loss, feed_dict={xs: x, ys: y})
            print loss_value
            prediction_value = sess.run(prediction, feed_dict={xs: x})
            lines = ax.plot(x, prediction_value, 'r-', lw=5)
            ax.lines.remove(lines[0])

            # plt.pause(0.01)

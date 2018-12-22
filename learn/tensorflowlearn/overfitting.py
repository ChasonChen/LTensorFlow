import os
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import addLayer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load
digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=3)

# define placeholder
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

# add layer
l1 = addLayer.add_layer(xs, 64, 100, 'layer_1', activation_func=tf.nn.tanh)
prediction = addLayer.add_layer(l1, 100, 10, 'layer_2', activation_func=tf.nn.softmax)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar("loss", cross_entropy)
step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

init = tf.global_variables_initializer()
merge = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)

    for i in range(1000):
        sess.run(step, feed_dict={xs: x_train, ys: y_train})
        if i % 50 == 0:
            result = sess.run(merge, feed_dict={xs: x, ys: y})

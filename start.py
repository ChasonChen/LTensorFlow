import tensorflow as tf
import pandas as pd

from feature_formater import format_data, get_word_vec, get_feature, add_vec2feature
from trainner import add_layer

df = get_feature()

word_vec = get_word_vec()

# xs = tf.constant(df['item'])
ys = tf.constant(add_vec2feature(df, word_vec), shape=[9999, 4 * 49])
#
# yp = tf.placeholder(tf.float32, [None, 1])
# xp = tf.placeholder(tf.float32, [None, 4 * 49])

# prediction = add_layer(xp, 4 * 49, 1, activation_function=None)
#
# loss_func = tf.reduce_mean(-tf.reduce_sum(yp * tf.log(prediction), reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss_func)
#
# init = tf.global_variables_initializer()
#
# print ('1')
# with tf.Session() as sess:
#     print ('2')
#     sess.run(init)
#     for i in range(1000):
#         print ('3')
#         sess.run(train_step, feed_dict={xp: xs, yp: ys})
#         if i % 50 == 0:
#             print ('4')
# print (sess.run(loss_func, feed_dict={xp: xs, yp: ys}))

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Variable ==============
# state = tf.Variable(0, name="counter")
# one = tf.constant(1)
#
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
#
# init = tf.global_variables_initializer()    # Call if define variable
#
# with tf.Session() as sess:
#     sess.run(init)  # Must call this.
#     for _ in range(30):
#         sess.run(update)
#         print sess.run(state)
#
#


# PlaceHolder =============
place1 = tf.placeholder(tf.float32)
place2 = tf.placeholder(tf.float32)

output = tf.multiply(place1, place2)
with tf.Session() as sess:
    print sess.run(output, feed_dict={place1: 3.0, place2: 4.0})

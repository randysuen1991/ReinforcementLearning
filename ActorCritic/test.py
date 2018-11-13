import tensorflow as tf

w1 = tf.get_variable('w1', shape=[3])
w2 = tf.get_variable('w2', shape=[3])

w3 = tf.get_variable('w3', shape=[3])
w4 = tf.get_variable('w4', shape=[3])

z1 = 3 * w1 + 2 * w2 + w3
z2 = -1 * w3 + w4

# grads = tf.gradients([z1, z2], [w1, w2, w3, w4], grad_ys=[tf.convert_to_tensor([2.,2.,3.]),
#                                                           tf.convert_to_tensor([3.,2.,4.])])
grads = tf.gradients([z1, z2], [w1, w2, w3, w4])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(grads))

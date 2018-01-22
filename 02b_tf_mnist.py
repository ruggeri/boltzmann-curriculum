import numpy as np
import tensorflow as tf

img_placeholder = tf.placeholder(
    tf.float32,
    shape = (None, 28, 28),
    name = "img"
)

rescaled_image = (img_placeholder - 128) / 128

flattened_img = tf.reshape(
    rescaled_image,
    (-1, 28 * 28)
)

weight_matrix1 = tf.Variable(
    tf.random_uniform(
        shape = (28 * 28, 128),
        minval = -1,
        maxval = +1
    ),
    name = 'weight_matrix1',
)

bias_vector1 = tf.Variable(
    tf.zeros(shape = (128,)),
    name = 'bias_vector1',
)

z1 = tf.matmul(flattened_img, weight_matrix1) + bias_vector1
h1 = tf.nn.sigmoid(z1)

weight_matrix2 = tf.Variable(
    tf.random_uniform(
        shape = (128, 1),
        minval = -1,
        maxval = +1
    ),
    name = 'weight_matrix2',
)

bias_vector2 = tf.Variable(
    tf.zeros(shape = (1,)),
    name = 'bias_vector2',
)

z2 = tf.matmul(h1, weight_matrix2) + bias_vector2
h2 = tf.nn.sigmoid(z2)

correct_y = tf.placeholder(tf.float32, shape = (None,))

eps = 1e-6
error = tf.reduce_mean(
    correct_y * -tf.log(h2 + eps)
    +
    (1 - correct_y) * -tf.log(1 - h2 + eps)
)

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate = 0.001
)

train_step = optimizer.minimize(error)

session = tf.Session()
session.run(tf.global_variables_initializer())

BATCH_SIZE = 32

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
for epoch_idx in range(1, 101):
    for batch_idx, batch_start in enumerate(range(0, x_train.shape[0], BATCH_SIZE)):
        x_batch = x_train[batch_start:(batch_start + BATCH_SIZE)]
        y_batch = y_train[batch_start:(batch_start + BATCH_SIZE)]

        _, e = session.run(
            [train_step, error],
            feed_dict = {
                img_placeholder: x_batch,
                correct_y: y_batch
            }
        )

        print(f'epoch: {epoch_idx:04d} | batch: {batch_idx:04d} | err: {e:0.1f}')

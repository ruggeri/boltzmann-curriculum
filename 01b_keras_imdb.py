from keras.datasets import imdb

TOP_N_WORDS = 1000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = TOP_N_WORDS)

import numpy as np

# Transform dataset from variable-length word sequences to a binary valued dense matrix.
new_x_train = np.zeros((len(x_train), TOP_N_WORDS + 1))
# We'll use a dummy column 0 to apply an intercept theta_0 to our model. It will always have value 1.
new_x_train[:, 0] = 1.0

for example_idx, word_sequence in enumerate(x_train):
    for word_idx in word_sequence:
        new_x_train[example_idx, word_idx] = 1

new_x_test = np.zeros((len(x_test), TOP_N_WORDS + 1))
new_x_test[:, 0] = 1.0
for example_idx, word_sequence in enumerate(x_test):
    for word_idx in word_sequence:
        new_x_test[example_idx, word_idx] = 1

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD

input_tensor = Input(shape = (new_x_train.shape[1],))
output_tensor = Dense(
    1,
    activation = 'sigmoid'
)(input_tensor)

m = Model(input_tensor, output_tensor)

optimizer = SGD(lr = 0.025)
m.compile(
    optimizer = optimizer,
    loss = 'binary_crossentropy',
    metrics = ['accuracy'],
)

m.fit(
    new_x_train,
    y_train,
    batch_size = 128,
    validation_data = (new_x_test, y_test),
    epochs = 100
)

from keras.datasets import boston_housing
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
import numpy as np

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Mean and variance normalize input data.
x_mean, x_stddev = x_train.mean(axis = 0), x_train.std(axis = 0)
x_train = (x_train - x_mean) / x_stddev
x_test = (x_test - x_mean) / x_stddev

y_mean, y_stddev = y_train.mean(), y_train.std()
y_train = (y_train - y_mean) / y_stddev
y_test = (y_test - y_mean) / y_stddev

input_tensor = Input(shape = (13,))
output_tensor = Dense(1, activation = 'linear')(input_tensor)

model = Model(input_tensor, output_tensor)

optimizer = SGD(lr = 0.1)

model.compile(
    loss = 'mse',
    optimizer = optimizer
)

model.fit(
    x_train,
    y_train,
    validation_data = (x_test, y_test),
    batch_size = 1024,
    epochs = 100
)

# How to interpret the MSE loss. The MSE is an estimate of *variance*,
# which is dispersion from the mean.
#
# Because we standardized y_train to have variance 1.0, that means
# that the MSE of guessing the mean value (which was itself set to
# zero) would be 1.0.
#
# If we train a model and have a test set MSE of 0.27, that means that
# the variance in output, *after* we have factored out the part of the
# output explained by the X variables, is 0.27.
#
# Put another way: the X variables explain 73% of the variance in
# house prices.

def calc_errors(y_train, y_predictions):
    absolute_errors = np.abs(y_train - y_predictions) * y_stddev
    mean_absolute_error = np.mean(absolute_errors)

    squared_errors = ((y_train - y_predictions) * y_stddev) ** 2
    mean_squared_error = np.mean(squared_errors)

    mean_absolute_percent_error = np.mean(
        np.abs(absolute_errors / y_mean)
    )

    return (mean_absolute_error, mean_squared_error, mean_absolute_percent_error)

(mean_absolute_error, mean_squared_error, mean_absolute_percent_error) = calc_errors(
    y_train,
    y_train.mean()
)
print(
    f"Baseline Mean Abs Err: {mean_absolute_error:0.1f} | "
    f"Baseline Mean Squared Err: {mean_squared_error:0.2f} | "
    f"Baseline Mean Abs %Err: {mean_absolute_percent_error:0.2f}"
)

(mean_absolute_error, mean_squared_error, mean_absolute_percent_error) = calc_errors(
    y_train,
    # model.predict gives us a (404, 1) matrix which won't play well
    # with our (404,) shape y_train. Thus we reshape the output to
    # (404,).
    model.predict(x_train).reshape((-1))
)
print(
    f"Model Mean Abs Err: {mean_absolute_error:0.1f} | "
    f"Model Mean Squared Err: {mean_squared_error:0.2f} | "
    f"Model Mean Abs %Err: {mean_absolute_percent_error:0.2f}"
)

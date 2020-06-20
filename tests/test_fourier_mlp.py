import pytest
import numpy as np
import scipy.stats as stats
import tensorflow as tf
import tf_fourier_features as tff

normal_dist = stats.norm(0, 1)


def test_fourier_mlp():
    tf.random.set_seed(0)

    model = tff.FourierFeatureMLP(units=256,
                                  final_units=3,
                                  activation='relu',
                                  final_activation='sigmoid',
                                  gaussian_projection=256,
                                  gaussian_scale=1.0)

    model.compile('adam', 'mse')

    x = tf.random.normal([100, 2])
    y = model(x)

    # we concat sine and cosine projections
    assert y.shape == (100, 3)

    y = tf.random.normal([100, 3])

    initial_loss = model.evaluate(x, y)
    model.fit(x, y, batch_size=10, epochs=1, verbose=0)
    final_loss = model.evaluate(x, y)

    assert initial_loss > final_loss


def test_fourier_mlp_with_scale():
    tf.random.set_seed(0)

    scale = 10.0
    model = tff.FourierFeatureMLP(units=256,
                                  final_units=3,
                                  activation='relu',
                                  final_activation='sigmoid',
                                  gaussian_projection=256,
                                  gaussian_scale=scale)

    model.compile('adam', 'mse')

    x = tf.random.normal([100, 2])
    y = model(x)

    # we concat sine and cosine projections
    assert y.shape == (100, 3)

    y = tf.random.normal([100, 3])

    initial_loss = model.evaluate(x, y)
    model.fit(x, y, batch_size=10, epochs=1, verbose=0)
    final_loss = model.evaluate(x, y)

    assert initial_loss > final_loss


if __name__ == '__main__':
    pytest.main(__file__)

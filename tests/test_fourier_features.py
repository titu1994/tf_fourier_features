import pytest
import numpy as np
import scipy.stats as stats
import tensorflow as tf
import tf_fourier_features as tff

normal_dist = stats.norm(0, 1)


def test_fourier_projection():
    tf.random.set_seed(0)

    ff_proj = tff.FourierFeatureProjection(gaussian_projection=256, gaussian_scale=1.0)

    x = tf.random.normal([100, 2])
    y = ff_proj(x)

    # we concat sine and cosine projections
    assert y.shape == (100, 256 * 2)


def test_fourier_projection_scaled():
    tf.random.set_seed(0)

    scale = 10.0
    ff_proj = tff.FourierFeatureProjection(gaussian_projection=256, gaussian_scale=scale)

    x = tf.random.normal([100, 2])
    y = ff_proj(x)

    # we concat sine and cosine projections
    assert y.shape == (100, 256 * 2)


def test_fourier_projection_identity():
    tf.random.set_seed(0)

    ff_proj = tff.FourierFeatureProjection(gaussian_projection=-1, gaussian_scale=1.0)

    x = tf.random.normal([100, 2])
    y = ff_proj(x)

    # we concat sine and cosine
    assert y.shape == (100, 4)


if __name__ == '__main__':
    pytest.main(__file__)


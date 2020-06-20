import tensorflow as tf
from typing import Optional
from tf_fourier_features import fourier_features


class FourierFeatureMLP(tf.keras.Model):

    def __init__(self, units: int, final_units: int, gaussian_projection: Optional[int],
                 activation: str = 'relu',
                 final_activation: str = "linear",
                 num_layers: int = 1,
                 gaussian_scale: float = 1.0,
                 use_bias: bool = True, **kwargs):
        """
        Fourier Feature Projection model from the paper
        [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://people.eecs.berkeley.edu/~bmild/fourfeat/).

        Used to create a multi-layer MLP with optional FourierFeatureProjection layer.

        Args:
            units: Number of hidden units in the intermediate layers.
            final_units: Number of hidden units in the final layer.
            activation: Activation in the hidden layers.
            final_activation: Activation function of the final layer.
            num_layers: Number of layers in the network.
            gaussian_projection: Projection dimension for the gaussian kernel in fourier feature
                projection layer. Can be None, negative or positive integer.
                If None, then fourier feature map layer is not used.
                If <=0, uses identity matrix (basic projection) without gaussian kernel.
                If >=1, uses gaussian projection matrix of specified dim.
            gaussian_scale: Scale of the gaussian kernel in fourier feature projection layer.
                Note: If the scale is too small, convergence will slow down and obtain poor results.
                If the scale is too large (>50), convergence will be fast but results will be grainy.
                Try grid search for scales in the range [10 - 50].
            use_bias: Boolean whether to use bias or not.

        # References:
            -   [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://people.eecs.berkeley.edu/~bmild/fourfeat/)
        """
        super().__init__(**kwargs)

        layers = []

        if gaussian_projection is not None:
            layers.append(fourier_features.FourierFeatureProjection(
                gaussian_projection=gaussian_projection,
                gaussian_scale=gaussian_scale,
                **kwargs
            ))

        for _ in range(num_layers - 1):
            layers.append(tf.keras.layers.Dense(units, activation=activation, use_bias=use_bias,
                                                bias_initializer='he_uniform', **kwargs))

        self.network = tf.keras.Sequential(layers)
        self.final_dense = tf.keras.layers.Dense(final_units, activation=final_activation,
                                                 use_bias=use_bias, bias_initializer='he_uniform', **kwargs)

    def call(self, inputs, training=None, mask=None):
        features = self.network(inputs)
        output = self.final_dense(features)
        return output


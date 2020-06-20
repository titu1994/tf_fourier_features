# Tensorflow Fourier Feature Mapping Networks
Tensorflow 2.0 implementation of Fourier Feature Mapping networks from the paper [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739).

# Installation

 - Pip install

```bash
$ pip install --upgrade tf_fourier_features
```

 - Pip install (test support)

```bash
$ pip install --upgrade tf_fourier_features[tests]
```

# Usage

```python
from tf_fourier_features import FourierFeatureProjection
from tf_fourier_features import FourierFeatureMLP

# You should use FourierFeatureProjection right after the input layer.
ip = tf.keras.layers.Input(shape=[2])
x = FourierFeatureProjection(gaussian_projection = 256, gaussian_scale = 1.0)(ip)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dense(3, activation='sigmoid')(x)
                                  
model = tf.keras.Model(inputs=ip, outputs=x)

# Or directly use the model class to build a multi layer Fourier Feature Mapping Network
model = FourierFeatureMLP(units=256, final_units=3, final_activation='sigmoid', num_layers=4,
                          gaussian_projection=256, gaussian_scale=10.0)
```

# Results on Image Inpainting task
A partial implementation of the image inpainting task is available as the `train_inpainting_fourier.py` and `eval_inpainting_fourier.py` scripts inside the `scripts` directory.

Weight files are made available in the repository under the `Release` tab of the project. Extract the weights and place the `checkpoints` folder at the scripts directory.

These weights generates the following output after 2000 epochs of training with batch size 8192 while using only 10% of the available pixels in the image during training phase.

<img src="https://github.com/titu1994/tf_fourier_features/blob/master/images/celtic_knot_10pct_kernel20.png?raw=true" height=100% width=100%>

------

If we train for using only 20% of the available pixels in the image during training phase - 

<img src="https://github.com/titu1994/tf_fourier_features/blob/master/images/celtic_knot_20pct_kernel20.png?raw=true" height=100% width=100%>

------

If we train for using only 30% of the available pixels in the image during training phase - .

<img src="https://github.com/titu1994/tf_fourier_features/blob/master/images/celtic_knot_30pct_kernel20.png?raw=true" height=100% width=100%>

# Citation

```
@misc{tancik2020fourier,
    title={Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains},
    author={Matthew Tancik and Pratul P. Srinivasan and Ben Mildenhall and Sara Fridovich-Keil and Nithin Raghavan and Utkarsh Singhal and Ravi Ramamoorthi and Jonathan T. Barron and Ren Ng},
    year={2020},
    eprint={2006.10739},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

# Requirements
 - Tensorflow 2.0+
 - Matplotlib to visualize eval result

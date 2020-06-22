import os
import pickle
from datetime import datetime
import tensorflow as tf
from tf_fourier_features.fourier_features_mlp import FourierFeatureMLP

SAMPLING_RATIO = 0.3
BATCH_SIZE = 8192
EPOCHS = 2000

IMAGE_SIZE = 800
IMAGE_EMBED = 8

img_filepath_1 = '../data/blue_flower.jpg'
img_filepath_2 = '../data/fur-style.jpg'
img_filepath_3 = '../data/celtic_spiral_knot.jpg'

images_paths = [img_filepath_1, img_filepath_2, img_filepath_3]
image_ground_truths = []

for img_filepath in images_paths:
    img_raw = tf.io.read_file(img_filepath)
    img_ground_truth = tf.io.decode_image(img_raw, channels=3, dtype=tf.float32)
    img_ground_truth = tf.image.resize(img_ground_truth, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.BICUBIC)
    image_ground_truths.append(img_ground_truth)

print("Decoded {} images of shape {}".format(len(image_ground_truths), image_ground_truths[0].shape))

rows, cols, channels = image_ground_truths[0].shape
pixel_count = rows * cols
sampled_pixel_count = int(pixel_count * SAMPLING_RATIO)


def build_train_tensors():
    img_mask_x = tf.random.uniform([sampled_pixel_count], maxval=rows, seed=0, dtype=tf.int32)
    img_mask_y = tf.random.uniform([sampled_pixel_count], maxval=cols, seed=1, dtype=tf.int32)

    img_mask_x = tf.expand_dims(img_mask_x, axis=-1)
    img_mask_y = tf.expand_dims(img_mask_y, axis=-1)

    img_mask_idx = tf.concat([img_mask_x, img_mask_y], axis=-1)

    train_images = []
    for image in image_ground_truths:
        img_train = tf.gather_nd(image, img_mask_idx, batch_dims=0)
        train_images.append(img_train)

    img_mask_x = tf.cast(img_mask_x, tf.float32) / rows
    img_mask_y = tf.cast(img_mask_y, tf.float32) / cols

    img_mask = tf.concat([img_mask_x, img_mask_y], axis=-1)
    image_masks = []
    image_contexts = []

    tf.random.set_seed(0)
    for ix in range(len(image_ground_truths)):
        img_context = tf.random.normal([1, IMAGE_EMBED])
        image_contexts.append(img_context)

        img_context = tf.broadcast_to(img_context, [img_mask.shape[0], IMAGE_EMBED])
        context_mask = tf.concat([img_mask, img_context], axis=-1)
        image_masks.append(context_mask)

    return image_masks, train_images, image_contexts


image_masks, train_images, image_contexts = build_train_tensors()
print("Number of masks and train images : ", len(image_masks))

img_mask = tf.concat(image_masks, axis=0)
img_train = tf.concat(train_images, axis=0)
print("Number of train samples : ", len(img_mask))

train_dataset = tf.data.Dataset.from_tensor_slices((img_mask, img_train))
train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Build model
model = FourierFeatureMLP(units=256, final_units=3, final_activation='sigmoid', num_layers=4,
                          gaussian_projection=256, gaussian_scale=20.0)

# instantiate model
_ = model(tf.zeros([1, 2 + IMAGE_EMBED]))

model.summary()

BATCH_SIZE = min(BATCH_SIZE, len(img_mask))
num_steps = int(len(img_mask) * EPOCHS / BATCH_SIZE)
print("Total training steps : ", num_steps)
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(0.0005, decay_steps=num_steps, end_learning_rate=0.0001,
                                                              power=2.0)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)  # Sum of squared error
model.compile(optimizer, loss=loss)

checkpoint_dir = 'checkpoints/multi_fourier_features/inpainting/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# Serailize the context vectors
with open(checkpoint_dir + 'image_contexts.pkl', 'wb') as f:
    pickle.dump(image_contexts, f)
print("Serialized the context embeddings !")


timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
logdir = os.path.join('../logs/multi_fourier_features/inpainting/', timestamp)

if not os.path.exists(logdir):
    os.makedirs(logdir)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + 'model', monitor='loss', verbose=0,
                                       save_best_only=True, save_weights_only=True, mode='min'),
    tf.keras.callbacks.TensorBoard(logdir, update_freq='batch', profile_batch=20)
]

model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks, verbose=2)


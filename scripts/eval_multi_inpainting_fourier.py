import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_fourier_features.fourier_features_mlp import FourierFeatureMLP

BATCH_SIZE = 8192

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

checkpoint_dir = 'checkpoints/multi_fourier_features/inpainting/'
checkpoint_path = checkpoint_dir + 'model'
if len(glob.glob(checkpoint_path + "*.index")) == 0:
    raise FileNotFoundError("Model checkpoint not found !")

# Load context vectors
with open(checkpoint_dir + 'image_contexts.pkl', 'rb') as f:
    image_contexts = pickle.load(f)

print("Loaded {} image contexts of size {}".format(len(image_contexts), image_contexts[0].shape[1]))


def build_eval_tensors():
    img_mask_x = tf.range(0, rows, dtype=tf.int32)
    img_mask_y = tf.range(0, cols, dtype=tf.int32)

    img_mask_x, img_mask_y = tf.meshgrid(img_mask_x, img_mask_y, indexing='ij')

    img_mask_x = tf.expand_dims(img_mask_x, axis=-1)
    img_mask_y = tf.expand_dims(img_mask_y, axis=-1)

    img_mask_x = tf.cast(img_mask_x, tf.float32) / rows
    img_mask_y = tf.cast(img_mask_y, tf.float32) / cols

    img_mask = tf.concat([img_mask_x, img_mask_y], axis=-1)
    img_mask = tf.reshape(img_mask, [-1, 2])

    image_masks = []
    for ix in range(len(image_ground_truths)):
        img_context = image_contexts[ix]
        img_context = tf.broadcast_to(img_context, [img_mask.shape[0], IMAGE_EMBED])
        context_mask = tf.concat([img_mask, img_context], axis=-1)
        image_masks.append(context_mask)

    eval_images = []
    for image in image_ground_truths:
        eval_images.append(image)

    return image_masks, eval_images


image_masks, eval_images = build_eval_tensors()

eval_datasets = []

for mask in image_masks:
    eval_dataset = tf.data.Dataset.from_tensor_slices((mask,))
    eval_dataset = eval_dataset.batch(BATCH_SIZE).cache()
    eval_dataset = eval_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    eval_datasets.append(eval_dataset)

# Build model
model = FourierFeatureMLP(units=256, final_units=3, final_activation='sigmoid', num_layers=4,
                          gaussian_projection=256, gaussian_scale=1.0)

# instantiate model
_ = model(tf.zeros([1, 2 + IMAGE_EMBED]))

# load checkpoint
model.load_weights(checkpoint_path).expect_partial()  # skip optimizer loading

model.summary()

# Predict pixels of the different images
output_images = []
for eval_dataset in eval_datasets:
    predicted_image = model.predict(eval_dataset, batch_size=BATCH_SIZE, verbose=1)
    predicted_image = predicted_image.reshape((rows, cols, channels))  # type: np.ndarray
    predicted_image = predicted_image.clip(0.0, 1.0)
    output_images.append(predicted_image)

fig, axes = plt.subplots(len(output_images), 2)

for ix, (ground_truth_img, predicted_img) in enumerate(zip(eval_images, output_images)):
    plt.sca(axes[ix, 0])
    gt_img = ground_truth_img.numpy()
    gt_img = gt_img.clip(0.0, 1.0)
    plt.imshow(gt_img)
    plt.title("Ground Truth Image")

    plt.sca(axes[ix, 1])
    plt.imshow(predicted_img)
    plt.title("Predicted Image")

fig.tight_layout()
plt.show()


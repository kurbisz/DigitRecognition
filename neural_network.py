from typing import Optional

import tensorflow as tf
import tensorflow_datasets as tfds


def model_for_larger_images(shape) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=shape + (1,))

    conv = tf.keras.layers.Conv2D(filters=4, kernel_size=3)(inputs)
    pooling = tf.keras.layers.GlobalAveragePooling2D()(conv)
    dense = tf.keras.layers.Dense(128, activation="relu")(pooling)
    feature = tf.keras.layers.Dense(10, activation="sigmoid")(dense)
    return tf.keras.Model(inputs, feature)


def preprocess_image(
    image: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [28, 28])
    return image, label


def train_and_test_mnist(
    epochs: int,
    layers: Optional[list[tf.keras.layers.Layer]] = None,
    model: Optional[tf.keras.Model] = None,
) -> tf.keras.callbacks.History:

    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = (
        ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(ds_info.splits["train"].num_examples)
        .batch(200)
        .prefetch(tf.data.AUTOTUNE)
    )

    ds_test = (
        ds_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(200)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    model = tf.keras.models.Sequential(layers) if layers else model

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model.fit(ds_train, epochs=epochs, validation_data=ds_test)


def test_layers():
    prefix = "[INFO] "

    shape = (28, 28)

    layers = [
        tf.keras.layers.Flatten(input_shape=shape),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(10, activation="sigmoid"),
    ]

    print(prefix, "TESTING DATA WITH 2 DENSE LAYERS")
    dense_network_history = train_and_test_mnist(epochs=10, layers=layers)
    print(dense_network_history.history)

    print(prefix, "TESTING DATA WITH 1 CONV, 1 POOL AND 2 DENSE LAYERS")
    model = model_for_larger_images(shape=shape)
    conv_network_history = train_and_test_mnist(epochs=10, model=model)
    print(conv_network_history.history)


if __name__ == "__main__":
    test_layers()

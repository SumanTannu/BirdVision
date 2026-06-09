"""Model-building utilities for EfficientNetB0 bird classification.

The model uses ImageNet-pretrained EfficientNetB0 as a frozen feature extractor
and adds a small custom classification head for the bird dataset. Keeping model
construction in this module makes training and evaluation scripts easier to
read and makes the architecture easier to modify later.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam


def build_augmentation(image_size: tuple[int, int]) -> tf.keras.Sequential:
    """Create image augmentation layers used during training.

    Args:
        image_size: Target `(height, width)` for resizing input images.

    Returns:
        A Keras `Sequential` layer that resizes images and applies random
        horizontal flips, rotations, zoom, and contrast changes. These
        transformations help the model generalize to bird photos with different
        framing, orientation, and lighting.
    """

    return tf.keras.Sequential(
        [
            layers.Resizing(*image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_efficientnetb0_model(num_classes: int, config) -> Model:
    """Build and compile the EfficientNetB0 transfer-learning model.

    Args:
        num_classes: Number of bird classes in the dataset. This controls the
            output layer size.
        config: `TrainConfig` instance containing image size, learning rate,
            dense layer sizes, and dropout settings.

    Returns:
        A compiled Keras `Model` ready for `model.fit`.

    Notes:
        EfficientNetB0 is loaded with `include_top=False` so the original
        ImageNet classifier is removed. The base model is frozen, which keeps
        training faster and reduces overfitting when the custom dataset is not
        huge. The custom dense head learns the bird-specific classification
        boundaries.
    """

    inputs = layers.Input(shape=(*config.image_size, 3))
    x = build_augmentation(config.image_size)(inputs)

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(*config.image_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    for units in config.hidden_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs, name="efficientnetb0_bird_classifier")
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

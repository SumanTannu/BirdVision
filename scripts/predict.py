"""Command-line single-image prediction entry point.

The script loads a saved Keras model and class-index JSON file, preprocesses one
image, and prints the most likely bird class with its confidence score.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from bird_classifier.config import TrainConfig
from bird_classifier.data import load_class_indices


def parse_args():
    """Parse command-line options for single-image prediction.

    Returns:
        An argparse namespace containing image path, model path, and class-index
        JSON path.
    """

    parser = argparse.ArgumentParser(description="Predict a bird class from one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=TrainConfig.output_dir / TrainConfig.model_name)
    parser.add_argument("--classes", type=Path, default=TrainConfig.output_dir / "class_indices.json")
    return parser.parse_args()


def main():
    """Load the model, classify one image, and print the result."""

    args = parse_args()
    model = tf.keras.models.load_model(args.model)
    index_to_label = load_class_indices(args.classes)

    image = tf.keras.preprocessing.image.load_img(args.image, target_size=TrainConfig.image_size)
    array = tf.keras.preprocessing.image.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.efficientnet.preprocess_input(array)

    probabilities = model.predict(array)[0]
    best_index = int(np.argmax(probabilities))
    print(f"Prediction: {index_to_label[best_index]}")
    print(f"Confidence: {probabilities[best_index]:.4f}")


if __name__ == "__main__":
    main()

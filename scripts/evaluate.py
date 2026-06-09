"""Command-line evaluation entry point for a saved bird classifier model.

The script reloads the trained model and class-index metadata, recreates the
same deterministic split from `train.csv`, evaluates the model, and writes a CSV
classification report.
"""

import argparse
import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bird_classifier.config import TrainConfig
from bird_classifier.data import build_train_dataframe, create_test_generator, load_class_indices, split_dataframe
from bird_classifier.evaluation import evaluate_model, prediction_labels, save_classification_report


def parse_args():
    """Parse command-line options for model evaluation.

    Returns:
        An argparse namespace containing dataset root, training CSV path, model
        path, and output directory overrides.
    """

    parser = argparse.ArgumentParser(description="Evaluate a trained bird classifier.")
    parser.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir)
    parser.add_argument("--train-csv", type=Path, default=TrainConfig.train_csv)
    parser.add_argument("--model", type=Path, default=TrainConfig.output_dir / TrainConfig.model_name)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    return parser.parse_args()


def main():
    """Run the saved-model evaluation workflow."""

    args = parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        output_dir=args.output_dir,
    )
    model = tf.keras.models.load_model(args.model)

    image_df = build_train_dataframe(config.data_dir, config.train_csv)
    _, test_df = split_dataframe(image_df, config.test_split, config.seed)
    test_images = create_test_generator(test_df, config)

    metrics = evaluate_model(model, test_images)
    index_to_label = load_class_indices(config.class_indices_path)
    predictions = prediction_labels(model, test_images, index_to_label)
    report = save_classification_report(list(test_df["Label"]), predictions, config.report_path)

    print(f"Test Loss: {metrics['loss']:.5f}")
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(report.head())


if __name__ == "__main__":
    main()

"""Command-line training entry point for the EfficientNetB0 bird classifier.

This script reads `birds-dataset/train.csv`, builds train/validation/test
generators, trains the EfficientNetB0 transfer-learning model, saves the model,
and writes evaluation artifacts to the output directory.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bird_classifier.callbacks import build_callbacks
from bird_classifier.config import TrainConfig
from bird_classifier.data import (
    build_train_dataframe,
    create_generators,
    save_class_indices,
    split_dataframe,
)
from bird_classifier.evaluation import (
    evaluate_model,
    prediction_labels,
    save_classification_report,
)
from bird_classifier.model import build_efficientnetb0_model
from bird_classifier.visualization import plot_history, plot_label_distribution


def parse_args():
    """Parse command-line options for training.

    Returns:
        An argparse namespace containing dataset root, CSV path, output path,
        epoch count, and batch size overrides.
    """

    parser = argparse.ArgumentParser(description="Train EfficientNetB0 bird classifier.")
    parser.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir)
    parser.add_argument("--train-csv", type=Path, default=TrainConfig.train_csv)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    return parser.parse_args()


def main():
    """Run the full training workflow from data loading to evaluation output."""

    args = parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    image_df = build_train_dataframe(config.data_dir, config.train_csv)
    plot_label_distribution(image_df, config.output_dir / "label_distribution.png")

    train_df, test_df = split_dataframe(image_df, config.test_split, config.seed)
    train_images, val_images, test_images = create_generators(train_df, test_df, config)
    save_class_indices(train_images.class_indices, config.class_indices_path)

    model = build_efficientnetb0_model(len(train_images.class_indices), config)
    history = model.fit(
        train_images,
        steps_per_epoch=len(train_images),
        validation_data=val_images,
        validation_steps=len(val_images),
        epochs=config.epochs,
        callbacks=build_callbacks(config),
    )
    model.save(config.output_dir / config.model_name)
    plot_history(history, config.output_dir / "training_curves.png")

    metrics = evaluate_model(model, test_images)
    index_to_label = {index: label for label, index in train_images.class_indices.items()}
    predictions = prediction_labels(model, test_images, index_to_label)
    report = save_classification_report(
        list(test_df["Label"]),
        predictions,
        config.report_path,
    )

    print(f"Test Loss: {metrics['loss']:.5f}")
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(report.head())


if __name__ == "__main__":
    main()

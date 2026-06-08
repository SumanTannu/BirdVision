"""Create a `predictions.csv` file from HV-AI-2024 `test.csv`.

The output file follows the expected challenge format:
`path,predicted_label,confidence_score`.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from bird_classifier.config import TrainConfig
from bird_classifier.data import (
    build_submission_dataframe,
    create_submission_generator,
    load_class_indices,
)


def format_label(label: str):
    """Return integer labels as integers while preserving non-numeric labels."""

    return int(label) if str(label).isdigit() else label


def parse_args():
    """Parse command-line options for submission generation."""

    parser = argparse.ArgumentParser(description="Create HV-AI predictions.csv.")
    parser.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir)
    parser.add_argument("--test-csv", type=Path, default=TrainConfig.test_csv)
    parser.add_argument("--model", type=Path, default=TrainConfig.output_dir / TrainConfig.model_name)
    parser.add_argument("--classes", type=Path, default=TrainConfig.output_dir / "class_indices.json")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    return parser.parse_args()


def main():
    """Load a trained model and write predictions for every test CSV row."""

    args = parse_args()
    config = TrainConfig(data_dir=args.data_dir, test_csv=args.test_csv)
    model = tf.keras.models.load_model(args.model)
    index_to_label = load_class_indices(args.classes)

    test_df = build_submission_dataframe(config.data_dir, config.test_csv)
    test_images = create_submission_generator(test_df, config)
    probabilities = model.predict(test_images)
    predicted_indices = probabilities.argmax(axis=1)
    confidence_scores = probabilities.max(axis=1)

    submission = pd.DataFrame(
        {
            "path": test_df["Path"],
            "predicted_label": [
                format_label(index_to_label[int(index)]) for index in predicted_indices
            ],
            "confidence_score": confidence_scores,
        }
    )
    submission.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()

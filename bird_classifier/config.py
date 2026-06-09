"""Configuration objects for the EfficientNetB0 bird classifier.

This module keeps training options in one place so the command-line scripts and
helper modules can share the same defaults. The values are intentionally simple
Python fields, which makes them easy to override from scripts or tests.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """Container for model training and output settings.

    Attributes:
        data_dir: Root directory of the HV-AI dataset. The expected layout is
            `HV-AI-2024/images/train`, `HV-AI-2024/images/test`,
            `HV-AI-2024/train.csv`, and `HV-AI-2024/test.csv`.
        train_csv: CSV file containing training image paths and class labels.
            The expected columns are `path`, `class`, and optionally `bbox`.
        test_csv: CSV file containing test image paths. The expected columns are
            `path` and optionally `bbox`.
        output_dir: Directory where trained models, reports, logs, plots, and
            class-index metadata are saved.
        model_name: Filename used for the final saved Keras model.
        image_size: Target `(height, width)` used by image generators and model
            inputs. EfficientNetB0 commonly uses 224x224 images.
        batch_size: Number of images loaded per training or evaluation batch.
        validation_split: Fraction of the training dataframe reserved for
            validation by `ImageDataGenerator`.
        test_split: Fraction of the full dataframe held out for final testing.
        seed: Random seed used for repeatable train/test splits and generators.
        learning_rate: Initial Adam optimizer learning rate.
        epochs: Maximum number of training epochs. Early stopping may finish
            training sooner.
        patience: Number of epochs with no validation-loss improvement before
            early stopping restores the best weights.
        reduce_lr_patience: Number of epochs with no validation-loss improvement
            before reducing the learning rate.
        label_smoothing: Softens one-hot labels during training so the model is
            less likely to become overconfident on the training set.
        dropout_rate: Dropout applied after the custom dense layer.
        hidden_units: Sizes of the custom dense layers placed on top of
            EfficientNetB0.
    """

    data_dir: Path = Path("HV-AI-2024")
    train_csv: Path = Path("HV-AI-2024/train.csv")
    test_csv: Path = Path("HV-AI-2024/test.csv")
    output_dir: Path = Path("outputs")
    model_name: str = "efficientnetb0_birds.keras"
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    validation_split: float = 0.2
    test_split: float = 0.2
    seed: int = 42
    learning_rate: float = 3e-4
    epochs: int = 150
    patience: int = 5
    reduce_lr_patience: int = 3
    label_smoothing: float = 0.1
    dropout_rate: float = 0.30
    hidden_units: tuple[int, ...] = (256,)

    @property
    def checkpoint_path(self) -> Path:
        """Return the path where the best validation checkpoint is saved."""

        return self.output_dir / "checkpoints" / self.model_name

    @property
    def class_indices_path(self) -> Path:
        """Return the JSON path used to persist class-name to index mappings."""

        return self.output_dir / "class_indices.json"

    @property
    def report_path(self) -> Path:
        """Return the CSV path used for the final classification report."""

        return self.output_dir / "classification_report.csv"

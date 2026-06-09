"""Dataset CSV loading, splitting, and image generator helpers.

The project expects the `birds-dataset` layout, where `train.csv` stores image
paths and class labels while images live under `images/train` and `images/test`.
The optional `bbox` column is kept out of model training because this classifier
uses full images, not cropped bounding boxes.
"""

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_train_dataframe(data_dir: Path, csv_path: Path) -> pd.DataFrame:
    """Build a labeled dataframe from `train.csv`.

    Args:
        data_dir: Dataset root directory, for example `birds-dataset`.
        csv_path: Path to the training CSV. It must contain `path` and `class`.

    Returns:
        A dataframe with `Filepath`, `Label`, and `Path` columns. `Filepath` is
        the actual local image path, `Label` is the class id as a string, and
        `Path` preserves the original CSV path for reporting.

    Raises:
        ValueError: If the CSV is missing required columns.
    """

    data_dir = Path(data_dir)
    df = pd.read_csv(csv_path)
    required_columns = {"path", "class"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing_columns)}")

    df = df.copy()
    df["Path"] = df["path"]
    df["Filepath"] = df["path"].apply(lambda path: str(data_dir / path))
    df["Label"] = df["class"].astype(str)
    return df[["Filepath", "Label", "Path"]]


def build_submission_dataframe(data_dir: Path, csv_path: Path) -> pd.DataFrame:
    """Build an unlabeled dataframe from `test.csv`.

    Args:
        data_dir: Dataset root directory, for example `birds-dataset`.
        csv_path: Path to the test CSV. It must contain `path`.

    Returns:
        A dataframe with `Filepath` and `Path` columns. `Path` preserves the CSV
        value required for the final prediction file.

    Raises:
        ValueError: If the CSV does not contain a `path` column.
    """

    data_dir = Path(data_dir)
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError(f"Missing column in {csv_path}: path")

    df = df.copy()
    df["Path"] = df["path"]
    df["Filepath"] = df["path"].apply(lambda path: str(data_dir / path))
    return df[["Filepath", "Path"]]


def split_dataframe(image_df: pd.DataFrame, test_split: float, seed: int):
    """Split image metadata into train and test dataframes.

    The split is stratified by label, which helps preserve class distribution in
    both train and test sets. This is important for bird datasets where many
    classes can have similar numbers of images.

    Args:
        image_df: Dataframe produced by `build_train_dataframe`.
        test_split: Fraction of rows to reserve for the test set.
        seed: Random seed for reproducible splitting.

    Returns:
        A tuple `(train_df, test_df)`.
    """

    return train_test_split(
        image_df,
        test_size=test_split,
        shuffle=True,
        random_state=seed,
        stratify=image_df["Label"],
    )


def create_generators(train_df: pd.DataFrame, test_df: pd.DataFrame, config):
    """Create train, validation, and test image generators.

    The train dataframe is split internally into training and validation subsets
    using `config.validation_split`. The test dataframe is kept separate and is
    never shuffled, so predictions stay aligned with `test_df`.

    Args:
        train_df: Training metadata dataframe.
        test_df: Final test metadata dataframe.
        config: `TrainConfig` instance with image size, batch size, seed, and
            validation split settings.

    Returns:
        A tuple `(train_images, val_images, test_images)` of Keras dataframe
        iterators.
    """

    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        validation_split=config.validation_split,
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    common_args = {
        "x_col": "Filepath",
        "y_col": "Label",
        "target_size": config.image_size,
        "color_mode": "rgb",
        "class_mode": "categorical",
        "batch_size": config.batch_size,
    }

    train_images = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        shuffle=True,
        seed=config.seed,
        subset="training",
        **common_args,
    )
    val_images = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        shuffle=True,
        seed=config.seed,
        subset="validation",
        **common_args,
    )
    test_images = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        shuffle=False,
        **common_args,
    )
    return train_images, val_images, test_images


def create_test_generator(test_df: pd.DataFrame, config):
    """Create a non-shuffled test image generator.

    This helper is used when a trained model is evaluated later without
    re-creating training and validation generators. Keeping `shuffle=False`
    ensures predicted labels can be compared directly against `test_df["Label"]`.

    Args:
        test_df: Dataframe containing image paths and labels for testing.
        config: `TrainConfig` instance with image size and batch size settings.

    Returns:
        A Keras dataframe iterator for evaluation.
    """

    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    return test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="Filepath",
        y_col="Label",
        target_size=config.image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=config.batch_size,
        shuffle=False,
    )


def create_submission_generator(test_df: pd.DataFrame, config):
    """Create a non-shuffled unlabeled test generator for challenge submission.

    Args:
        test_df: Dataframe produced by `build_submission_dataframe`.
        config: `TrainConfig` instance with image size and batch size settings.

    Returns:
        A Keras dataframe iterator that yields images only.
    """

    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    return test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="Filepath",
        y_col=None,
        target_size=config.image_size,
        color_mode="rgb",
        class_mode=None,
        batch_size=config.batch_size,
        shuffle=False,
    )


def save_class_indices(class_indices: dict[str, int], path: Path) -> None:
    """Save Keras class-index mappings to JSON.

    Keras generators map class names to numeric output indices. Saving that
    mapping is required for later prediction, because model outputs are numeric
    probabilities and must be translated back into bird class names.

    Args:
        class_indices: Mapping from class name to numeric class index.
        path: JSON output path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(class_indices, file, indent=2, sort_keys=True)


def load_class_indices(path: Path) -> dict[int, str]:
    """Load class-index mappings and invert them for prediction.

    Args:
        path: JSON file produced by `save_class_indices`.

    Returns:
        A dictionary mapping numeric class index to class name.
    """

    with Path(path).open("r", encoding="utf-8") as file:
        class_indices = json.load(file)
    return {index: label for label, index in class_indices.items()}

"""Evaluation helpers for trained bird classification models.

This module keeps metrics and prediction-label conversion separate from the
training script. That makes it possible to reuse the same evaluation logic after
training or when loading a saved model later.
"""

import pandas as pd
from sklearn.metrics import classification_report


def evaluate_model(model, test_images):
    """Evaluate a Keras model on a test image generator.

    Args:
        model: Trained Keras model.
        test_images: Non-shuffled Keras dataframe iterator for test images.

    Returns:
        A dictionary with `loss` and `accuracy` values.
    """

    loss, accuracy = model.evaluate(test_images, verbose=0)
    return {"loss": loss, "accuracy": accuracy}


def prediction_labels(model, images, index_to_label: dict[int, str]):
    """Predict class names for all images in a generator.

    Args:
        model: Trained Keras model.
        images: Keras image iterator to predict on.
        index_to_label: Mapping from numeric output index to class name.

    Returns:
        A list of predicted bird class names in generator order.
    """

    predictions = model.predict(images)
    predicted_indices = predictions.argmax(axis=1)
    return [index_to_label[index] for index in predicted_indices]


def save_classification_report(y_true, y_pred, output_path):
    """Create and save a scikit-learn classification report.

    Args:
        y_true: Ground-truth class names.
        y_pred: Predicted class names.
        output_path: CSV file path for the report.

    Returns:
        A pandas dataframe containing precision, recall, f1-score, and support
        values for each class plus aggregate rows.
    """

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)
    return report_df

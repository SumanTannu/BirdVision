"""Training callback factory for the bird classifier.

Callbacks handle training-time behaviors that are not part of the model graph:
saving the best model, stopping when validation loss stops improving, reducing
the learning rate on plateaus, and writing TensorBoard logs.
"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


def build_callbacks(config):
    """Create the callback list used by `model.fit`.

    Args:
        config: `TrainConfig` instance containing output paths and patience
            values.

    Returns:
        A list of Keras callbacks:
        `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, and
        `TensorBoard`.
    """

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = config.output_dir / "training_logs" / "bird_classification"

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(config.checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=config.reduce_lr_patience,
            min_lr=1e-6,
        ),
        TensorBoard(log_dir=str(log_dir)),
    ]

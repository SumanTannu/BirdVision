"""Visualization helpers for dataset analysis and model interpretation.

These functions are optional helpers for saving charts and Grad-CAM outputs.
They do not change model training behavior, but they make it easier to inspect
class balance, training progress, and which image regions influenced a
prediction.
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_label_distribution(image_df, output_path=None, top_n=20):
    """Plot the most frequent labels in the dataset.

    Args:
        image_df: Dataframe with a `Label` column.
        output_path: Optional image path where the plot should be saved.
        top_n: Number of most frequent classes to display.

    Returns:
        The matplotlib figure object.
    """

    label_counts = image_df["Label"].value_counts()[:top_n]
    plt.figure(figsize=(18, 6))
    plt.bar(label_counts.index, label_counts.values)
    plt.title(f"Top {top_n} Bird Classes")
    plt.xlabel("Class")
    plt.ylabel("Images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    return plt.gcf()


def plot_history(history, output_path=None):
    """Plot training and validation accuracy/loss curves.

    Args:
        history: Keras `History` object returned by `model.fit`.
        output_path: Optional image path where the figure should be saved.

    Returns:
        The matplotlib figure object containing accuracy and loss subplots.
    """

    metrics = history.history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(metrics["accuracy"], label="train")
    axes[0].plot(metrics["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(metrics["loss"], label="train")
    axes[1].plot(metrics["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    return fig


def get_img_array(img_path, size):
    """Load an image as a batched numpy array.

    Args:
        img_path: Path to an image file.
        size: Target image size expected by the model.

    Returns:
        A numpy array with shape `(1, height, width, channels)`.
    """

    image = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(image)
    return np.expand_dims(array, axis=0)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv", pred_index=None):
    """Generate a Grad-CAM heatmap for a model prediction.

    Args:
        img_array: Batched image array, usually from `get_img_array`.
        model: Trained Keras model.
        last_conv_layer_name: Name of the convolution layer used for Grad-CAM.
        pred_index: Optional class index to explain. If omitted, the model's
            highest-scoring class is used.

    Returns:
        A normalized 2D numpy heatmap with values between 0 and 1.
    """

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    """Overlay a Grad-CAM heatmap on the original image and save it.

    Args:
        img_path: Path to the original image.
        heatmap: 2D heatmap returned by `make_gradcam_heatmap`.
        cam_path: Output path for the Grad-CAM visualization.
        alpha: Strength of the heatmap overlay. Higher values make the heatmap
            more visible.

    Returns:
        The output path where the Grad-CAM image was saved.
    """

    image = tf.keras.preprocessing.image.load_img(img_path)
    image = tf.keras.preprocessing.image.img_to_array(image)

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + image
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    cam_path.parent.mkdir(parents=True, exist_ok=True)
    superimposed_img.save(cam_path)
    return cam_path

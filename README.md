# Bird Classification with EfficientNetB0

This project trains a bird species classifier using TensorFlow/Keras and the
EfficientNetB0 transfer-learning model. It is set up for the `HV-AI-2024`
dataset layout, where labels are stored in `train.csv` and the test images are
listed in `test.csv`.

EfficientNetB0 is used as a pretrained feature extractor. The model starts with
ImageNet weights, removes the original ImageNet classification head, and adds a
new bird-species classification head for your dataset.

## Structure

```text
.
+-- bird_classifier/
|   +-- __init__.py
|   +-- callbacks.py
|   +-- config.py
|   +-- data.py
|   +-- evaluation.py
|   +-- model.py
|   +-- visualization.py
+-- scripts/
|   +-- __init__.py
|   +-- evaluate.py
|   +-- predict.py
|   +-- create_submission.py
|   +-- train.py
+-- original EfficientNetB0 launcher .py file
+-- requirements.txt
+-- README.md
```

### Important Files

- `bird_classifier/config.py` - default paths, image size, epochs, batch size,
  learning rate, dropout, and output filenames.
- `bird_classifier/data.py` - scans the image folders, creates train/test
  splits, and builds Keras image generators.
- `bird_classifier/model.py` - builds the EfficientNetB0 transfer-learning
  model.
- `bird_classifier/callbacks.py` - creates early stopping, checkpoint,
  learning-rate reduction, and TensorBoard callbacks.
- `bird_classifier/evaluation.py` - evaluates the model and saves the
  classification report.
- `bird_classifier/visualization.py` - creates label distribution plots,
  training curves, and Grad-CAM visualizations.
- `scripts/train.py` - main training command.
- `scripts/evaluate.py` - evaluates a saved model.
- `scripts/predict.py` - predicts the bird class for one image.
- `scripts/create_submission.py` - creates `predictions.csv` for the HV-AI test
  set.
- Original EfficientNetB0 launcher `.py` file - simple launcher that runs the
  training script. The recommended command is still `python scripts/train.py`.

## Dataset Format

Your dataset should be arranged like this:

```text
HV-AI-2024/
+-- images/
|   +-- train/
|   +-- test/
+-- train.csv
+-- test.csv
```

The training CSV should contain image paths and class labels:

```csv
path,class,bbox
images/train/1_2.jpg,0,139.0 30.0 153.0 264.0
```

The test CSV should contain image paths without labels:

```csv
path,bbox
images/test/test_81.jpg,241.0 113.0 202.0 257.0
```

The `bbox` column is ignored by this classifier. The model trains on the full
image.

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

If TensorFlow installation is slow or fails, install it separately according to
your Python version and system.

## Train the Model

Train using the default settings:

```bash
python scripts/train.py
```

This uses:

```text
data directory: HV-AI-2024
train CSV:      HV-AI-2024/train.csv
```

Train with fewer epochs:

```bash
python scripts/train.py --epochs 20
```

Change batch size:

```bash
python scripts/train.py --batch-size 16
```

If your dataset is in a different folder:

```bash
python scripts/train.py --data-dir "path/to/HV-AI-2024" --train-csv "path/to/HV-AI-2024/train.csv"
```

You can also run the launcher file if you kept the original filename. For most
usage, `scripts/train.py` is the cleaner command.

## Outputs

Training creates an `outputs/` folder containing:

```text
outputs/
+-- checkpoints/
|   +-- efficientnetb0_birds.keras
+-- class_indices.json
+-- classification_report.csv
+-- efficientnetb0_birds.keras
+-- label_distribution.png
+-- training_curves.png
+-- training_logs/
```

Important output files:

- `efficientnetb0_birds.keras` - final saved model.
- `checkpoints/efficientnetb0_birds.keras` - best validation checkpoint.
- `class_indices.json` - mapping between class names and model output indexes.
- `classification_report.csv` - precision, recall, f1-score, and support.
- `label_distribution.png` - chart of the most common labels.
- `training_curves.png` - training/validation accuracy and loss curves.

## Evaluate a Saved Model

After training, evaluate the saved model:

```bash
python scripts/evaluate.py
```

If your model is stored somewhere else:

```bash
python scripts/evaluate.py --model "outputs/efficientnetb0_birds.keras"
```

## Predict One Image

Predict a single bird image:

```bash
python scripts/predict.py "path/to/bird.jpg"
```

The command prints the predicted class name and confidence score.

## Create `predictions.csv`

After training, create the final prediction file for `HV-AI-2024/test.csv`:

```bash
python scripts/create_submission.py
```

This writes:

```text
predictions.csv
```

with this format:

```csv
path,predicted_label,confidence_score
images/test/test_81.jpg,42,0.91
```

If your model or test CSV is in another location:

```bash
python scripts/create_submission.py --test-csv "path/to/test.csv" --model "outputs/efficientnetb0_birds.keras" --output "predictions.csv"
```

## Model Summary

The model pipeline is:

```text
Input image
-> augmentation layers
-> EfficientNetB0 pretrained backbone
-> Dense layer
-> Dropout
-> Dense layer
-> Dropout
-> Softmax bird-class output
```

By default, the EfficientNetB0 base model is frozen. This means only the custom
classification head is trained. This is faster and usually works well as a
first transfer-learning approach.

## Configuration

Most settings are in `bird_classifier/config.py`:

```python
image_size = (224, 224)
batch_size = 32
learning_rate = 3e-4
epochs = 150
dropout_rate = 0.20
hidden_units = (512,)
```

Update these values if you want to tune training behavior.

## Notes

- Keep `shuffle=False` for test/submission generators so predictions align
  correctly with CSV rows.
- Keep `class_indices.json` with the saved model. Prediction needs it to convert
  numeric model outputs back into bird class labels.
- Large bird datasets can take a long time to train on CPU. GPU runtime is
  recommended.
- A 3-epoch run is mainly a smoke test. For useful accuracy, train for more
  epochs, for example `python scripts/train.py --epochs 30`.

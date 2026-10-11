# Tuberculosis Detection from Chest X-Rays

A convolutional neural network that classifies chest X-ray images as tuberculosis or normal, trained with TensorFlow/Keras.
The classifier reached about 90% accuracy and was deployed in a Django web app.

## Models

| Notebook | Model |
|---|---|
| [`code/M1.ipynb`](code/M1.ipynb) | Custom CNN: Conv2D (32 filters) → max pooling → dense layers, trained with RMSprop |
| [`code/M2.ipynb`](code/M2.ipynb) | VGG16 |
| [`code/M3.ipynb`](code/M3.ipynb) | LeNet |

Images are resized to 256×256 and loaded with Keras `ImageDataGenerator`.

## Repository

- `code/`: training notebooks for the three models
- `Report.docx`: project report

## Stack

Python · TensorFlow / Keras · NumPy · Pandas · Matplotlib · Django

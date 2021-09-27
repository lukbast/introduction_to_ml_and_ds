import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models
import tensorflow_hub as hub
import datetime
import os

IMAGE_SIZE = (224, 224)


def img_to_tensor(img_path: str) -> tf.Tensor:
    """
    Utility function that makes converting images to Tensor easier.
    :param img_path: relative path to an image
    :return: Tensor created from given image
    """
    global IMAGE_SIZE
    # Read in an image file
    img = tf.io.read_file(img_path)
    # turn the image into a tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Convert the colour channel values from 0-255 to 0-1 values
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize the image to desired value
    img = tf.image.resize(img, size=IMAGE_SIZE)
    return img


def create_tensor_tuple(image_path: str, label):
    """
    Creates Tensor tuples, image will be converted to Tensor and resized to given shape if provided.
    :param image_path: relative path to an image
    :param label: Label of an image
    :return: Tuple containing Tensor with an image nd its label
    """
    img = img_to_tensor(image_path)
    return img, label


def create_data_batches(x, y=None, batch_size: int = 32,
                        valid_data: bool = False,
                        test_data: bool = False,
                        img_size: tuple = None):
    """
    Create a batches of given size.
    Before it, images are loaded, turned into Tensors and resized
    if img_size was provided.
    :param x: dataset of image paths
    :param y: dataset of labels
    :param batch_size: size of an batch
    :param valid_data: Set it to true if creating a validation batch
    :param test_data: Set it to true if creating a test batch
    :param img_size: image will be resized to this shape after converting it to an Tensor
    :return: Batch
    """
    globals()["IMAGE_SIZE"] = img_size
    # If the data is a test dataset, we most likely don't have labels
    if test_data and y is None:
        print("Creating test batches...")

        data = tf.data.Dataset.from_tensor_slices(tf.constant(x))  # only file paths (no labels)
        data_batches = data.map(img_to_tensor).batch(batch_size)
        return data_batches
    # if the data is a validation dataset, there's no need to shuffle it
    elif valid_data or test_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
        data_batches = data.map(create_tensor_tuple).batch(batch_size)
        return data_batches
    else:
        print("Creating Training data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x),
                                                   tf.constant(y)))
        # It's faster to shuffle paths names instead of images
        data = data.shuffle(buffer_size=len(x))
        data = data.map(create_tensor_tuple)
        data_batches = data.batch(batch_size=batch_size)
        return data_batches


def show_images(images, labels, unique_labels, n_of_img: int) -> None:
    """
    Displays a plot of n images
    :param n_of_img: number of images to display, best when it gives integer sqrt like 25
    :param images: Batch of images
    :param labels: Batch of labels
    :param unique_labels: list of unique labels. before turning into booleans
    :return: None
    """

    # Setup the figure
    plt.figure(figsize=(10, 10))
    # Loop trough images
    for i in range(n_of_img):
        # Create subplots (5 rows, 5 cols)
        # noinspection PyUnusedLocal
        ax = plt.subplot(int(math.sqrt(n_of_img)), int(math.sqrt(n_of_img)), i + 1)
        # Display an image
        plt.imshow(images[i])
        # Add the image label as the title
        plt.title(unique_labels[labels[i].argmax()])
        # turn the grid lines
        plt.axis("off")


def create_model(input_shape, output_shape, model_url: str):
    """
    Build an Keras sequential deep learning model.
    :param input_shape: input shape in format (batch, height, width, colour channels)
    :param output_shape: number of unique labels
    :param model_url: url of model to use in transfer learning
    :return: compiled and build model
    """
    print("Building model with", model_url)
    # Setup the model layers
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url),  # Layer 1 (input layer)
        layers.Dense(units=output_shape, activation="softmax")  # Layer 2 (output)
    ])

    # compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="Adam",
        metrics="accuracy"
    )

    model.build(input_shape)

    return model


def create_tensorboard_callback():
    # get path to directory in which logs are being held
    logdir = os.path.join("/content/drive/MyDrive/dog-vision-colab/logs",
                          datetime.datetime.utcnow().strftime("%Y/%m/%d-%H:%M:%S"))
    return callbacks.TensorBoard(logdir)


def train_model(train_data, valid_data,
                input_shape, output_shape,
                model_url: str, num_of_epochs: int,
                callbacks_: tuple):
    # Create a model
    model = create_model(input_shape, output_shape, model_url)

    # Fit the model to de data passing callbacks we created
    model.fit(x=train_data,
              epochs=num_of_epochs,
              validation_data=valid_data,
              validation_freq=1,
              callbacks=callbacks_
              )

    return model


def get_pred_label(pred_prob, unique_labels):
    """
    Turns an array of prediction probabilities into a label
    :param unique_labels: list of unique labels
    :param pred_prob: Array of prediction probabilities for single sample
    :return: predicted label
    """

    return unique_labels[np.argmax(pred_prob)]


def unbatch_dataset(batch, unique_labels):
    """
    :param batch: TensorFlow's Batch
    :param unique_labels: list of unique labels
    :return:tuple containing arrays of images and labels
    """

    images = []
    labels = []

    for img, lab in batch.unbatch().as_numpy_iterator():
        images.append(img),
        labels.append(unique_labels[np.argmax(lab)])

    return images, labels


def plot_pred(pred_probs, images, labels, unique_labels, n=0):
    """
    Plots image with predicted and true label, title of image is green
    if prediction was true.
    :param pred_probs: prediction probabilities fro set of data
    :param labels: list of true labels
    :param images: list of images
    :param unique_labels: list of unique labels
    :param n: index if image to print
    :return: None
    """

    pred_prob, true_label, img = pred_probs[n], labels[n], images[n]

    # Get the pred_label
    pred_label = get_pred_label(pred_prob, unique_labels)

    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    # plot image & remove ticks
    plt.imshow(img)
    plt.axis("off")

    # Change plot title to be predicted, probability of prediction and truth label
    plt.title("Pred: {}   Prob {:2.0f}%   True: {}".format(pred_label,
                                                           np.max(pred_prob) * 100,
                                                           true_label),
              color=color)


def plot_pred_conf(preds, true_labels, unique_labels, n=1):
    """
    Plots top 10 highest prediction confidences along with the truth label for sample n.
    :param preds: list of predictions made by model
    :param true_labels: list of true labels
    :param unique_labels: list of unique labels for data set
    :param n: index if a sample to plot
    :return: None
    """

    pred_probs, true_label = preds[n], true_labels[n]

    # Find the top 10 predictions confidence indexes
    top_10_indexes = pred_probs.argsort()[-10:][::-1]

    # Find the top 10 prediction confidence values
    top_10_pred_values = pred_probs[top_10_indexes]

    # Find the top 10 prediction labels
    top_10_labels = unique_labels[top_10_indexes]

    # Setup plot
    top_plot = plt.bar(np.arange(len(top_10_labels)),
                       top_10_pred_values,
                       color="gray"
                       )
    plt.xticks(np.arange(len(top_10_labels)),
               labels=top_10_labels,
               rotation="vertical"
               )

    # change color of true label
    if np.isin(true_label, top_10_labels):
        top_plot[np.argmax(top_10_labels == true_label)].set_color("green")


def save_model(model, dir_path: str, suffix: str = None) -> str:
    """
    Saves a model in given directory. Model will be named with UTC
    time of saving the model in format %Y-%m-%d--%H:%M:%S"with suffix appended at the end.
    :param model: TensorFlow Keras model
    :param dir_path: Path of directory where model will be saved
    :param suffix: Optional, string that will be appended to name of the model
    :return: Saved model's path
    """
    # Create a model directory path name with current time

    model_dir = os.path.join(dir_path, datetime.datetime.utcnow().strftime("%Y-%m-%d--%H:%M:%S"))
    model_path = f"{model_dir}-{suffix}.h5"
    print(f"Saving model to: {model_path}...")

    model.save(model_path)

    return model_path


def load_model(model_path: str):
    """
    Loads TensorFlow's Keras model from specified path
    :param model_path: Path to saved model
    :return: Loaded TensorFlow's Keras model
    """

    print(f"Loading saved model from: {model_path}")
    model = models.load_model(model_path,
                              custom_objects={"KerasLayer": hub.KerasLayer})

    return model

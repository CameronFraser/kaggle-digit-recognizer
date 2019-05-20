"""
Digit Recognition classes for TF 2.0
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU, Dropout
from tensorflow.keras import Model
from toolz.functoolz import pipe


class DigitRecognitionModel(Model):
    """
        The DigitRecognitionModel class contains the layers for a deep neural network.
        The tf.function decorator compiles the function into a graph which will execute on the GPU
    """

    def __init__(self):
        super(DigitRecognitionModel, self).__init__()
        self.conv1 = Conv2D(64, 3)
        self.maxpool = MaxPool2D(2, padding="SAME")
        self.relu = ReLU()
        self.conv2 = Conv2D(32, 3)
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation="relu")
        self.dropout = Dropout(0.5)
        self.fc2 = Dense(10, activation="softmax")

    @tf.function
    def call(self, X):
        """
            tf.function: Defines the forward pass

            Args:
                x (ndarray): The input images

            Returns:
                predictions (ndarray): Probability distribution over all classes
        """
        return pipe(
            X,
            self.conv1,
            self.maxpool,
            self.relu,
            self.conv2,
            self.maxpool,
            self.relu,
            self.flatten,
            self.fc1,
            self.dropout,
            self.fc2,
        )


class DigitRecognition:
    """
        The DigitRecognition class handles the training, validation, and testing pipelines.
        The tf.function decorator compiles the function into a graph which will execute on the GPU

        Args:
            model (DigitRecognitionModel): Model containing network layers
            optimizer (Optimizer): Optimizer class (Adam in this case)
            loss_object (function): Loss function for calculating loss
            train_data (tf.data.Dataset): Batched training data
            valid_data (tf.data.Dataset): Batched validation data
    """

    def __init__(self, model, optimizer, loss_object, train_data, valid_data):
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )

        self.valid_loss = tf.keras.metrics.Mean(name="valid_loss")
        self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="valid_accuracy"
        )

        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="test_accuracy"
        )

        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss_object

        self.train_data = train_data
        self.valid_data = valid_data

    @tf.function
    def train_step(self, images, labels):
        """
            tf.function: Trains on supplied images & labels

            Args:
                images: Batch of images to make predictions on
                labels: Batch of labels to calculate loss and accuracy
        """
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def valid_step(self, images, labels):
        """
            tf.function: Validates model performance at the end of every epoch

            Args:
                images: Batch of images to make predictions on
                labels: Batch of labels to calculate loss and accuracy
        """
        predictions = self.model(images)
        loss = self.loss_object(labels, predictions)

        self.valid_loss(loss)
        self.valid_accuracy(labels, predictions)

    @tf.function
    def test_predict(self, images):
        """
            tf.function: Makes predictions on input images

            Args:
                images: Images to make predictions on (batch these if you get OOM error)

            Returns:
                predictions (ndarray): Probability distribution over all classes for each image
        """
        predictions = self.model(images)
        return predictions

    @tf.function
    def test_score(self, labels, predictions):
        """
            tf.function: Calculate loss and accuracy

            Args:
                labels (ndarray): Known labels for images
                predictions (ndarray): Predicted labels
        """
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def fit(self, epochs=10):
        """
            Iterates over number of epochs and number of batches
            making calls to training and validation step methods

            Args:
                epochs (int): The number of training epochs to run
        """
        for epoch in range(epochs):
            for images, labels in self.train_data:
                self.train_step(images, labels)

            for images, labels in self.valid_data:
                self.valid_step(images, labels)

            template = (
                "Epoch {}, "
                "Train Loss: {}, "
                "Train Accuracy: {}, "
                "Validation Loss: {}, "
                "Validation Accuracy: {}"
            )

            print(
                template.format(
                    epoch + 1,
                    self.train_loss.result(),
                    self.train_accuracy.result() * 100,
                    self.valid_loss.result(),
                    self.valid_accuracy.result() * 100,
                )
            )

    def predict(self, images):
        """
            Make a call to prediction tf.function

            Args:
                images: Images to make predictions on

            Returns:
                predictions: Probability distribution over all classes for each image
        """
        predictions = self.test_predict(images)
        return predictions

    def score(self, labels, predictions):
        """
            Make a call to scoring tf.function

            Args:
                labels: Known labels
                predictions: Predicated labels

            Return:
                test_loss: Loss calculcated from supplied input
                test_accuracy: Accuracy calculated from supplied input
        """
        self.test_score(labels, predictions)
        return self.test_loss.result(), self.test_accuracy.result()

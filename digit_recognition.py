import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU, Dropout
from tensorflow.keras import Model
from toolz.functoolz import pipe

class DigitRecognitionModel(Model):
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
        
    def call(self, X):
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
            self.fc2
        )

class DigitRecognition:
    def __init__(self, model, optimizer, loss_object, train_data, valid_data, test_data):
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        
        self.valid_loss = tf.keras.metrics.Mean(name="valid_loss")
        self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_accuracy")
        
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
        self.test_top_k = lambda y_true, y_pred: tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, name="test_top_k")
        
        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss_object
        
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
    
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
    
    @tf.function
    def valid_step(self, images, labels):
        predictions = self.model(images)
        t_loss = self.loss_object(labels, predictions)

        self.valid_loss(t_loss)
        self.valid_accuracy(labels, predictions)
    
    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        self.test_top_k(labels, predictions)

    def fit(self, epochs=10):
        for epoch in range(epochs):
            for images, labels in self.train_data:
                self.train_step(images, labels)

            for images, labels in self.valid_data:
                self.valid_step(images, labels)

            template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
            print(
                template.format(
                    epoch + 1,
                    self.train_loss.result(),
                    self.train_accuracy.result() * 100,
                    self.valid_loss.result(),
                    self.valid_accuracy.result() * 100
                )
            )
    
    def predict(self):
        pass
    
    def score(self):
        pass
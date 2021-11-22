import os

import flwr as fl
import tensorflow as tf
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os.path

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)

        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["binary_accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_binary_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}

        
class CentralizedTrainer:
        def __init__(self, model, training_dataset, testing_dataset, batch_size = 32, epochs = 1, validation_split = 0.1):
            self.model = model
            self.x_train, self.y_train = training_dataset
            self.x_test, self.y_test = testing_dataset
            self.epochs = epochs
            self.validation_split = validation_split
            self.batch_size = batch_size
            self.run_model()
            
        def run_model(self):
            self.model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics= tf.keras.metrics.BinaryAccuracy())
            self.history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size= self.batch_size, validation_split=self.validation_split)

fl_dataset = None

def ids_iot_2020_datasets():
    global fl_dataset
    if fl_dataset is not None:
        return fl_dataset
    fname = "combinedcsvs"
    if not os.path.isfile(fname + ".csv"):
        with zipfile.ZipFile(fname + ".zip", "r") as zip_ref:
            zip_ref.extractall()
        
    dataset = pd.read_csv(fname + ".csv")
    list_protocol = dataset.drop_duplicates(subset = ['protocol'])['protocol']
    protocol = { row: idx + 1 for idx, row in enumerate(list_protocol) }
    dataset.protocol = [protocol[item] for item in dataset.protocol]
    features = pd.DataFrame(dataset.iloc[:, 3:30].values)
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(features)
    my_imputer = SimpleImputer(missing_values = np.nan,
                            strategy ='mean')
    X = pd.DataFrame(my_imputer.fit_transform(X))
    y = dataset.iloc[:, 30].values
    X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size = 0.05)
    fl_dataset = (X_training, y_training), (X_testing, y_testing), X.shape[1]
    return fl_dataset

def build_model_with_parameters(batch_size = 32, epochs = 100, validation_split = 0.1):
    (x_train, y_train), (x_test, y_test), shape = ids_iot_2020_datasets()
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=14, activation='relu', input_shape=(shape,)))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    return model, (x_train, y_train), (x_test, y_test), shape, batch_size, epochs, validation_split
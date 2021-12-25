import argparse
import os

import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import flwr as fl

import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import aux
# Make TensorFlow logs less verbose



def main() -> None:

    # Load and compile Keras model
    model, (x_train, y_train), (x_test, y_test), *_ = aux.build_model_with_parameters(epochs=1)

    model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=tf.keras.metrics.BinaryAccuracy())

    # Load a subset of CIFAR-10 to simulate the local data partition
    # (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = aux.FlowerClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("[::]:8080", client=client)


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    (x_train, y_train), (x_test, y_test), shape = aux.ids_iot_2020_datasets()
    assert idx in range(10)
    return (
               x_train[idx * 5000: (idx + 1) * 5000],
               y_train[idx * 5000: (idx + 1) * 5000],
           ), (
               x_test[idx * 1000: (idx + 1) * 1000],
               y_test[idx * 1000: (idx + 1) * 1000],
           )


if __name__ == "__main__":
    main()

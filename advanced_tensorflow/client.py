import argparse
import os

from random import randrange
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

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 15), required=True)
    args = parser.parse_args()

    model, *_ = aux.build_model_with_parameters(epochs=1, categorical=True, nrows=100)

    model.compile(optimizer='Adam', loss=tf.keras.losses.CategoricalCrossentropy(),
                       metrics=['accuracy'])

    # Load a subset of CIFAR-10 to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = aux.FlowerClient(model, x_train, y_train, x_test, y_test, validation_split=0.01)
    fl.client.start_numpy_client("[::]:8080", client=client)



def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    (x_train, y_train), (x_test, y_test), _ = aux.ids_iot_2020_datasets(rowsperdataset=100, categorical=True)
    assert idx in range(13)
    if idx > 9:
        idx = randrange(10)
    return (
               x_train[idx * 35: (idx + 1) * 35],
               y_train[idx * 35: (idx + 1) * 35],
           ), (
               x_test[idx * 15: (idx + 1) * 15],
               y_test[idx * 15: (idx + 1) * 15],
           )


if __name__ == "__main__":
    main()

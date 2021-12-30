import argparse
import os

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import tensorflow as tf

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import aux

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--instance", type=int, choices=range(1, 21), required=True)
args = parser.parse_args()

model, train, test, shape, batch_size, epochs, validation_split = aux.build_model_with_parameters(epochs=100)

trainer = aux.CentralizedTrainer(model, train, test, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split, instance=args.instance)


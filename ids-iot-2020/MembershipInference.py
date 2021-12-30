import argparse
import os

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, CarliniLInfMethod
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.utils import to_categorical
import tensorflow.python.ops.numpy_ops.np_config as np_config
np_config.enable_numpy_behavior()

import privacy_evaluator.models.torch.dcti.dcti as torch_dcti
import privacy_evaluator.models.tf.dcti.dcti as tf_dcti

from privacy_evaluator.datasets.tf.cifar10 import TFCIFAR10
from privacy_evaluator.datasets.torch.cifar10 import TorchCIFAR10

from privacy_evaluator.classifiers.classifier import Classifier

from privacy_evaluator.attacks.membership_inference.black_box import MembershipInferenceBlackBoxAttack
from privacy_evaluator.attacks.membership_inference import MembershipInferenceAttackAnalysis

from privacy_evaluator.attacks.membership_inference import MembershipInferenceAttackAnalysis

from privacy_evaluator.attacks.membership_inference.data_structures.attack_input_data import AttackInputData
from privacy_evaluator.attacks.membership_inference.data_structures.slicing import Slicing

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import aux

model, train, test, shape, batch_size, epochs, validation_split = aux.build_model_with_parameters(
    epochs=3, categorical=True)

(x_train, y_train), (x_test, y_test) = train, test

trainer = aux.CentralizedTrainer(model, train, test, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split, categorical=True)

vulnerable_model = trainer.model

classifier = Classifier(vulnerable_model, tf.keras.losses.CategoricalCrossentropy(), 2, (shape,))

attack = MembershipInferenceBlackBoxAttack(classifier)

print(attack)

attack.fit(x_train[:100], y_train[:100], x_test[:100], y_test[:100])
attack.attack(x_train[:100], y_train[:100])

output = attack.attack_output(
    x_train[:100],
    y_train[:100],
    x_train,
    y_train,
    x_test,
    y_test,
    np.ones((100,))
)

print(output.to_json())


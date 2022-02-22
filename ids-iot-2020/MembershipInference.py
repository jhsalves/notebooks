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
from privacy_evaluator.attacks.membership_inference.black_box_rule_based import MembershipInferenceBlackBoxRuleBasedAttack
from privacy_evaluator.attacks.membership_inference.label_only_decision_boundary import MembershipInferenceLabelOnlyDecisionBoundaryAttack
from privacy_evaluator.attacks.membership_inference import MembershipInferenceAttackAnalysis

from privacy_evaluator.attacks.membership_inference import MembershipInferenceAttackOnPointBasis
from privacy_evaluator.attacks.membership_inference import MembershipInferencePointAnalysis

from privacy_evaluator.attacks.membership_inference.data_structures.attack_input_data import AttackInputData
from privacy_evaluator.attacks.membership_inference.data_structures.slicing import Slicing

from privacy_evaluator.output.user_output_privacy_score import UserOutputPrivacyScore

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import aux

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--sample", type=int, choices=range(0, 21), required=True)
args = parser.parse_args()

sample = str(args.sample)

model, train, test, shape, batch_size, epochs, validation_split = aux.build_model_with_parameters(nrows=10000,
                                                                                                  epochs=30,
                                                                                                  categorical=True)

(x_train, y_train), (x_test, y_test) = train, test
"""
trainer = aux.CentralizedTrainer(model, train, test, epochs=epochs, batch_size=batch_size,
                                 validation_split=0.1, categorical=True)

trainer.model.save("centralizado100epocascombined.h5")

target_model = trainer.model
"""
def blackbox_attack(target_model):
    aux.write_line(sample, "blackbox_attack")
    classifier = get_classifier(target_model)

    attack = MembershipInferenceBlackBoxAttack(classifier)

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
    aux.write_line(sample, "output: " + str(output.to_json()))
    analysis = MembershipInferenceAttackAnalysis(
        MembershipInferenceBlackBoxAttack,
        AttackInputData(
            x_train[:100],
            y_train[:100],
            x_test[:100],
            y_test[:100]
        )
    )

    slicing = Slicing(
        entire_dataset=True,
        by_class=False,
        by_classification_correctness=False
    )

    result = analysis.analyse(
        classifier,
        np.concatenate((x_train[:100], x_test[:100])),
        np.concatenate((y_train[:100], y_test[:100])),
        np.concatenate((np.ones(len(x_train[:100])), np.zeros(len(x_test[:100])))),
        slicing
    )

    print("\n".join((str(r) for r in result)))
    aux.write_line(sample, "\n result: ".join((str(r) for r in result)))


def get_classifier(target_model):
    classifier = Classifier(target_model, tf.keras.losses.CategoricalCrossentropy(), 2, (shape,))
    return classifier


def rule_based_attack(target_model):
    aux.write_line(sample, "rule_based_attack")
    classifier = get_classifier(target_model)
    attack = MembershipInferenceBlackBoxRuleBasedAttack(classifier)
    attack.attack(x_train, y_train)
    output = attack.attack_output(
        x_train[:100],
        y_train[:100],
        x_train,
        y_train,
        x_test,
        y_test,
        np.ones((len(y_train[:100]),))
    )

    aux.write_line(sample, "output: " + str(output))
    output.to_dict()

    analysis = MembershipInferenceAttackAnalysis(
        MembershipInferenceBlackBoxRuleBasedAttack,
        AttackInputData(
            x_train[:100],
            y_train[:100],
            x_test[:100],
            y_test[:100]
        )
    )

    slicing = Slicing(
        entire_dataset=True,
        by_class=False,
        by_classification_correctness=False
    )

    result = analysis.analyse(
        classifier,
        np.concatenate((x_train[:100], x_test[:100])),
        np.concatenate((y_train[:100], y_test[:100])),
        np.concatenate((np.ones(len(x_train[:100])), np.zeros(len(x_test[:100])))),
        slicing
    )

    print("\n".join((str(r) for r in result)))
    aux.write_line(sample, "\n result: ".join((str(r) for r in result)))


def label_only_decision_boundary(target_model):
    aux.write_line(sample, "label_only_decision_boundary")
    target_model = get_classifier(target_model)
    attack = MembershipInferenceLabelOnlyDecisionBoundaryAttack(target_model)
    attack.fit(x_train[:100], y_train[:100], x_test[:20], y_test[:20], max_iter=5, max_eval=5, init_eval=5)
    attack.attack(x_train[500:520], y_train[500:520])
    output = attack.attack_output(
        x_train[500:520],
        y_train[500:520],
        x_train[:520],
        y_train[:520],
        x_test[:520],
        y_test[:520],
        np.ones((20,))
    )

    aux.write_line(sample, "output: " + str(output))


    analysis = MembershipInferenceAttackAnalysis(
        MembershipInferenceLabelOnlyDecisionBoundaryAttack,
        AttackInputData(
            x_train[:100],
            y_train[:100],
            x_test[:100],
            y_test[:100]
        )
    )

    slicing = Slicing(
        entire_dataset=True,
        by_class=False,
        by_classification_correctness=False
    )

    result = analysis.analyse(
        target_model,
        np.concatenate((x_train[:100], x_test[500:600])),
        np.concatenate((y_train[:100], y_test[500:600])),
        np.concatenate((np.ones(len(x_train[:100])), np.zeros(len(x_test[:100])))),
        slicing
    )

    aux.write_line(sample, "\n result: ".join((str(r) for r in result)))


centralizado = tf.keras.models.load_model("centralizado100epocas250000.h5")
federado = tf.keras.models.load_model("client0.h5")

aux.write_line(sample, "----------------------------*BEGIN SAMPLE " + sample + "*----------------------------------")
titleCentralized = "---------------CENTRALIZADO-----------------"
print(titleCentralized)
aux.write_line(sample, titleCentralized)
blackbox_attack(centralizado)

titleFederated = "---------------FEDERADO----------------------"
aux.write_line(sample, titleFederated)
print(titleFederated)
blackbox_attack(federado)

aux.write_line(sample, "--------------------------------------------------------------")
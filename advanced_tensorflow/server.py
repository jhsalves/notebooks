from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf
import sys
import os
import argparse
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import aux


def main() -> None:

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--instance")
    args = parser.parse_args()

    aux.write_distributed(["\n", "\n", f"---------------------------- {args.instance} ------------------------------"])

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model, _, (x_test, y_test), *_ = aux.build_model_with_parameters(epochs=1)
    model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=tf.keras.metrics.BinaryAccuracy())

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.8,
        fraction_eval=0.8,
        min_fit_clients=10,
        min_eval_clients=10,
        min_available_clients=10,
        eval_fn=get_eval_fn(model, x_test, y_test),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": 9}, strategy=strategy)


def get_eval_fn(model, x_test, y_test):
    """Return an evaluation function for server-side evaluation."""

    x_val, y_val = x_test, y_test

    # The `evaluate` function will be called after every round
    def evaluate(
            weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        aux.write_distributed([f"loss: {loss}, accuracy: {accuracy}"])
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 10
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()

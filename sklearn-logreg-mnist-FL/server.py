import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict

import argparse


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_mnist()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        print (f'parm ={parameters[1]}')
        loss = log_loss(y_test, model.predict_proba(X_test))
        print (f'loss = {loss}')
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
DEFAULT_STRATEGY= "FedAvg"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        help=f"strategy (default: {DEFAULT_STRATEGY})",
    )
    args = parser.parse_args()


    model = LogisticRegression()
    utils.set_initial_params(model)

    if args.strategy == "FedAvg":
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=2,
            evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=fit_round,
        )
    elif args.strategy == "FedAdagrad":
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=0.3,
            fraction_evaluate=0.3,
            min_available_clients=2,
            evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=fit_round,
            initial_parameters=fl.common.ndarrays_to_parameters(
            utils.get_model_parameters(model)),
    )
        
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=20),
    )

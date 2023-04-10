import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict, Callable

import argparse

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else 2,  #
    }
    return config

# on_fit_config_fn can be used to pass arbitrary configuration values 
# from server to client, and potentially change these values each round
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(server_round: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "learning_rate": str(0.001),
            "batch_size": str(32),
        }
        return config

    return fit_config

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
def main(args) -> None:
    # The logistic regression model is defined and initialized with utils.set_initial_params()
    model = LogisticRegression()
    utils.set_initial_params(model)

    strategy = None
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
            #on_fit_config_fn=fit_round,
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.ndarrays_to_parameters(
            utils.get_model_parameters(model)),
        )
    elif args.strategy == "FedAdam":
        strategy = fl.server.strategy.FedAdam(
            fraction_fit=0.3,
            fraction_evaluate=0.3,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=fit_config,
            accept_failures=0,
            initial_parameters=fl.common.ndarrays_to_parameters(
            utils.get_model_parameters(model)),
        )
    elif args.strategy == "FedYogi":
        strategy = fl.server.strategy.FedYogi(
            fraction_fit=0.3,
            fraction_evaluate=0.3,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=fit_config,
            accept_failures=0,
            initial_parameters=fl.common.ndarrays_to_parameters(
            utils.get_model_parameters(model)),
        )
    else:
        args.strategy = "Not defined"
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=2,
            evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=get_on_fit_config_fn()
        )
        

    print (f'{args.strategy}')
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(args.num_rounds),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        help=f"strategy (default: {DEFAULT_STRATEGY})",
    )
    parser.add_argument("--num-rounds", default=1, type=int)
    parser.add_argument("--num-clients", default=2, type=int)
    args = parser.parse_args()
    main(args)


    
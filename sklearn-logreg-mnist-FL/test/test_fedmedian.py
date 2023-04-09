####################################
from unittest.mock import MagicMock

from numpy import array, float32
from flwr.common import (
    Code,
    FitRes,
    NDArrays,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


from flwr.server.client_proxy import ClientProxy
from flwr.server.grpc_server.grpc_client_proxy import GrpcClientProxy
from flwr.server.strategy.fedmedian import FedMedian


def test_aggregate_fit() -> None:
    """Tests if FedMedian is aggregating correctly."""
    # Prepare
    previous_weights: NDArrays = [array([0.1, 0.1, 0.1, 0.1], dtype=float32)]
    strategy = FedMedian(
        initial_parameters=ndarrays_to_parameters(previous_weights),
    )
    param_0: Parameters = ndarrays_to_parameters(
        [array([0.2, 0.2, 0.2, 0.2], dtype=float32)]
    )
    param_1: Parameters = ndarrays_to_parameters(
        [array([1.0, 1.0, 1.0, 1.0], dtype=float32)]
    )
    param_2: Parameters = ndarrays_to_parameters(
        [array([0.5, 0.5, 0.5, 0.5], dtype=float32)]
    )
    bridge = MagicMock()
    client_0 = GrpcClientProxy(cid="0", bridge=bridge)
    client_1 = GrpcClientProxy(cid="1", bridge=bridge)
    client_2 = GrpcClientProxy(cid="2", bridge=bridge)
    results: List[Tuple[ClientProxy, FitRes]] = [
        (
            client_0,
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=param_0,
                num_examples=5,
                metrics={},
            ),
        ),
        (
            client_1,
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=param_1,
                num_examples=5,
                metrics={},
            ),
        ),
        (
            client_2,
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=param_2,
                num_examples=5,
                metrics={},
            ),
        ),
    ]
    expected: NDArrays = [array([0.5, 0.5, 0.5, 0.5], dtype=float32)]

    # Execute
    actual_aggregated, _ = strategy.aggregate_fit(
        server_round=1, results=results, failures=[]
    )
    if actual_aggregated:
        actual_list = parameters_to_ndarrays(actual_aggregated)
        actual = actual_list[0]

    print (f"actual={actual} expected={expected[0]}")
    assert (actual == expected[0]).all()

def main():
    test_aggregate_fit()

main()
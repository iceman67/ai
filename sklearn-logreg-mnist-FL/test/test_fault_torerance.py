"""FaultTolerantFedAvg tests."""


from typing import List, Optional, Tuple, Union
from unittest.mock import MagicMock

from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    NDArrays,
    Parameters,
    Status,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.fault_tolerant_fedavg import FaultTolerantFedAvg


def test_aggregate_fit_no_results_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.1)
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    expected: Optional[Parameters] = None

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_fit_no_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.1)
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = [Exception()]
    expected: Optional[Parameters] = None

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_fit_not_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.5)
    results: List[Tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                Status(code=Code.OK, message="Success"),
                Parameters(tensors=[], tensor_type=""),
                1,
                {},
            ),
        )
    ]
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = [
        Exception(),
        Exception(),
    ]
    expected: Optional[Parameters] = None

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_fit_just_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.5)
    results: List[Tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                Status(code=Code.OK, message="Success"),
                Parameters(tensors=[], tensor_type=""),
                1,
                {},
            ),
        )
    ]
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = [Exception()]
    expected: Optional[NDArrays] = []

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual
    assert parameters_to_ndarrays(actual) == expected


def test_aggregate_fit_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.99)
    results: List[Tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                Status(code=Code.OK, message="Success"),
                Parameters(tensors=[], tensor_type=""),
                1,
                {},
            ),
        )
    ]
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    expected: Optional[NDArrays] = []

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual
    assert parameters_to_ndarrays(actual) == expected


def test_aggregate_evaluate_no_results_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.1)
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    expected: Optional[float] = None

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_no_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.1)
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = [
        Exception()
    ]
    expected: Optional[float] = None

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_not_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.5)
    results: List[Tuple[ClientProxy, EvaluateRes]] = [
        (
            MagicMock(),
            EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=2.3,
                num_examples=1,
                metrics={},
            ),
        )
    ]
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = [
        Exception(),
        Exception(),
    ]
    expected: Optional[float] = None

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_just_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.5)
    results: List[Tuple[ClientProxy, EvaluateRes]] = [
        (
            MagicMock(),
            EvaluateRes(
                Status(code=Code.OK, message="Success"),
                loss=2.3,
                num_examples=1,
                metrics={},
            ),
        )
    ]
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = [
        Exception()
    ]
    expected: Optional[float] = 2.3

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.99)
    results: List[Tuple[ClientProxy, EvaluateRes]] = [
        (
            MagicMock(),
            EvaluateRes(
                Status(code=Code.OK, message="Success"),
                loss=2.3,
                num_examples=1,
                metrics={},
            ),
        )
    ]
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    expected: Optional[float] = 2.3

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)
    print (f"actual={actual} expected={expected}")

    # Assert
    assert actual == expected


def main():
    test_aggregate_evaluate_no_failures()

main()
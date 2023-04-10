
from typing import List, Tuple

import numpy as np

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


def test_aggregate() -> None:
    """Test aggregate function."""

    # Prepare
    weights0_0 = np.array([[1, 2, 3], [4, 5, 6]])
    weights0_1 = np.array([7, 8, 9, 10])
    weights1_0 = np.array([[1, 2, 3], [4, 5, 6]])
    weights1_1 = np.array([7, 8, 9, 10])
    results = [([weights0_0, weights0_1], 1), ([weights1_0, weights1_1], 2)]

    expected = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([7, 8, 9, 10])]

  
    # Execute
    actual = aggregate(results)
    
    print (f'actual ={actual}, expected ={expected}')

    # Assert
    np.testing.assert_equal(expected, actual)  # type: ignore


def test_weighted_loss_avg_single_value() -> None:
    """Test weighted loss averaging."""
    # Prepare
    results: List[Tuple[int, float]] = [(5, 0.5)]
    expected = 0.5

    # Execute
    actual = weighted_loss_avg(results)

    # Assert
    assert expected == actual


def test_weighted_loss_avg_multiple_values() -> None:
    """Test weighted loss averaging."""
    # Prepare
    results: List[Tuple[int, float]] = [(1, 2.0), (2, 1.0), (1, 2.0)]
    expected = 1.5

    # Execute
    actual = weighted_loss_avg(results)
    print (f'actual ={actual}, expected ={expected}')

    # Assert
    assert expected == actual


test_aggregate()
test_weighted_loss_avg_multiple_values()


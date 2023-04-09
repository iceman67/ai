import numpy as np
import pytest

from flwr.common.parameter import bytes_to_ndarray, ndarray_to_bytes


def test_serialisation_deserialisation() -> None:
    """Test if the np.ndarray is identical after (de-)serialization."""
    arr = np.array([[1, 2], [3, 4], [5, 6]])

    arr_serialized = ndarray_to_bytes(arr)
    arr_deserialized = bytes_to_ndarray(arr_serialized)

    # Assert deserialized array is equal to original
    np.testing.assert_equal(arr_deserialized, arr)  # type: ignore

    # Test false positive
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        np.testing.assert_equal(arr_deserialized, np.ones((3, 2)))  # type: ignore

    print(f'{arr_deserialized}, {arr}')
test_serialisation_deserialisation()
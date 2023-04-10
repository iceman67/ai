from typing import Union, cast

from flwr.common import typing
from flwr.proto import transport_pb2 as pb2

from flwr.common.serde import (
    scalar_from_proto,
    scalar_to_proto,
    status_from_proto,
    status_to_proto,
)


def test_serialisation_deserialisation() -> None:
    """Test if the np.ndarray is identical after (de-)serialization."""

    # Prepare
    scalars = [True, b"bytestr", 3.14, 9000, "Hello"]

    for scalar in scalars:
        # Execute
        scalar = cast(Union[bool, bytes, float, int, str], scalar)
        serialized = scalar_to_proto(scalar)
        actual = scalar_from_proto(serialized)

        print (f'{actual}, {scalar}')

        # Assert
        assert actual == scalar
    


def test_status_to_proto() -> None:
    """Test status message (de-)serialization."""

    # Prepare
    code_msg = pb2.Code.OK
    status_msg = pb2.Status(code=code_msg, message="Success")

    code = typing.Code.OK
    status = typing.Status(code=code, message="Success")

    # Execute
    actual_status_msg = status_to_proto(status=status)

    # Assert
    assert actual_status_msg == status_msg


def test_status_from_proto() -> None:
    """Test status message (de-)serialization."""

    # Prepare
    code_msg = pb2.Code.OK
    status_msg = pb2.Status(code=code_msg, message="Success")

    code = typing.Code.OK
    status = typing.Status(code=code, message="Success")

    # Execute
    actual_status = status_from_proto(msg=status_msg)

    # Assert
    assert actual_status ==  status

test_serialisation_deserialisation()
import base64
import io
import json
import socket
import threading

import numpy as np
import pytest

from concertina_eval_client.protocol import (
    PolicyClient,
    ProtocolError,
    ServerError,
    decode,
    encode,
    recv_message,
    send_message,
)


@pytest.mark.parametrize("dtype", ["uint8", "int32", "int64", "float16", "float32", "float64", "bool"])
def test_arrays_round_trip_bitwise(dtype):
    array = (np.arange(24).reshape(2, 3, 4) % 7).astype(dtype)
    decoded = decode(encode({"x": [array, {"y": array[0]}]}))
    for value, expected in ((decoded["x"][0], array), (decoded["x"][1]["y"], array[0])):
        assert value.dtype == expected.dtype and value.shape == expected.shape
        assert value.tobytes() == expected.tobytes()
        assert value.flags.writeable


def test_non_contiguous_and_scalar_values():
    array = np.arange(12, dtype=np.float32).reshape(3, 4).T
    decoded = decode(encode({"a": array, "i": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True)}))
    np.testing.assert_array_equal(decoded["a"], array)
    assert decoded["i"] == 3 and decoded["f"] == 0.5 and decoded["b"] is True


def _npy(array: np.ndarray, allow_pickle: bool = False) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=allow_pickle)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _npz() -> str:
    buffer = io.BytesIO()
    np.savez(buffer, x=np.zeros(3))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_arrays_travel_as_npy():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    message = json.loads(encode({"x": array}))
    assert message == {"x": {"__ndarray__": _npy(array)}}


@pytest.mark.parametrize(
    "array_payload",
    [
        _npy(np.array([object(), 1], dtype=object), allow_pickle=True),  # pickled object array
        _npy(np.array(["a", "b"])),  # unicode
        _npy(np.zeros(2, dtype=[("a", "f4"), ("b", "i4")])),  # structured
        _npy(np.zeros(2, dtype=np.complex64)),
        _npz(),
        _npy(np.zeros(8, dtype=np.float32))[:-8],  # truncated data
        base64.b64encode(b"not an npy file").decode(),
        "@@@@",
        123,
    ],
    ids=["object", "unicode", "structured", "complex", "npz", "truncated", "garbage", "base64", "type"],
)
def test_unsafe_or_malformed_arrays_are_rejected(array_payload):
    with pytest.raises(ProtocolError):
        decode(json.dumps({"res": {"__ndarray__": array_payload}}).encode())


def test_array_payload_with_extra_keys_is_rejected():
    payload = {"__ndarray__": _npy(np.zeros(2)), "other": 1}
    with pytest.raises(ProtocolError):
        decode(json.dumps(payload).encode())


def test_invalid_json_is_rejected():
    with pytest.raises(ProtocolError):
        decode(b"not json")


def test_object_arrays_cannot_be_sent():
    with pytest.raises(TypeError):
        encode(np.array([object()]))


def test_framing_over_fragmented_stream():
    left, right = socket.socketpair()
    with left, right:
        message = {"obs": np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)}
        threading.Thread(target=send_message, args=(left, message)).start()
        received = recv_message(right)
        assert received["obs"].tobytes() == message["obs"].tobytes()
        left.close()
        assert recv_message(right) is None


def test_oversized_message_is_rejected():
    left, right = socket.socketpair()
    with left, right:
        send_message(left, {"x": "a" * 1000})
        with pytest.raises(ProtocolError):
            recv_message(right, max_bytes=100)


def test_truncated_message_is_an_error():
    left, right = socket.socketpair()
    with right:
        left.sendall((100).to_bytes(4, "big") + b"{}")
        left.close()
        with pytest.raises(ProtocolError):
            recv_message(right)


def test_client_raises_server_errors_and_reconnects():
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]

    def serve():
        for reply in ({"error": "boom", "traceback": "tb"}, {"res": [1, 2]}):
            connection, _ = listener.accept()
            with connection:
                recv_message(connection)
                send_message(connection, reply)

    thread = threading.Thread(target=serve)
    thread.start()
    with listener, PolicyClient(port=port, timeout=10, connect_timeout=10) as client:
        with pytest.raises(ServerError, match="boom"):
            client.call("predict", {})
        assert client.call("predict", {}) == [1, 2]
    thread.join()

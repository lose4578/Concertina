"""Wire protocol between simulation clients and the Concertina policy server.

A message is a 4-byte big-endian length followed by that many bytes of UTF-8 JSON. A NumPy array travels as
``{"__ndarray__": <base64 of the array in NumPy's .npy format>}``; only boolean, integer and floating-point arrays
are accepted, and pickled data never is. Requests are ``{"cmd": <str>, "obs": <payload or null>}``; responses are
``{"res": <payload>}`` or ``{"error": <str>, "traceback": <str>}``.

Only NumPy and the standard library are used, so that simulation environments (some on Python 3.8) can import it.
"""

from __future__ import annotations

import base64
import io
import json
import socket
import time
from typing import Any

import numpy as np

HEADER_BYTES = 4
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
ARRAY_KEY = "__ndarray__"
# Plain numeric/boolean arrays only: decoding never creates object arrays.
_ALLOWED_DTYPE_KINDS = frozenset("biuf")


class ProtocolError(RuntimeError):
    """Malformed or oversized message."""


class ServerError(RuntimeError):
    """The server reported an error for a request."""


def _encode_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind not in _ALLOWED_DTYPE_KINDS:
            raise TypeError(f"Cannot send arrays of dtype {obj.dtype}")
        buffer = io.BytesIO()
        np.save(buffer, obj, allow_pickle=False)
        return {ARRAY_KEY: base64.b64encode(buffer.getvalue()).decode("ascii")}
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not serializable")


def _decode_hook(obj: dict) -> Any:
    if ARRAY_KEY not in obj:
        return obj
    if len(obj) != 1 or not isinstance(obj[ARRAY_KEY], str):
        raise ProtocolError("An array payload must be a single base64 string")
    try:
        # `read_array` accepts only the .npy format (no .npz archives) and, without pickles, only plain dtypes.
        array = np.lib.format.read_array(
            io.BytesIO(base64.b64decode(obj[ARRAY_KEY], validate=True)), allow_pickle=False
        )
    except (ValueError, TypeError, OSError, EOFError) as exc:
        raise ProtocolError(f"Invalid array payload: {exc}") from exc
    if array.dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise ProtocolError(f"Array dtype {array.dtype} is not allowed")
    return array


def encode(payload: Any) -> bytes:
    return json.dumps(payload, default=_encode_default).encode("utf-8")


def decode(data: bytes) -> Any:
    try:
        return json.loads(data.decode("utf-8"), object_hook=_decode_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Invalid JSON message: {exc}") from exc


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    """Read exactly ``size`` bytes; None if the peer closed the connection before the first byte."""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            if remaining == size:
                return None
            raise ProtocolError("Connection closed in the middle of a message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, payload: Any) -> None:
    body = encode(payload)
    sock.sendall(len(body).to_bytes(HEADER_BYTES, "big") + body)


def recv_message(sock: socket.socket, max_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> Any:
    """The next message, or None if the peer closed the connection between messages."""
    header = _recv_exact(sock, HEADER_BYTES)
    if header is None:
        return None
    length = int.from_bytes(header, "big")
    if length > max_bytes:
        raise ProtocolError(f"Message of {length} bytes exceeds the limit of {max_bytes}")
    body = _recv_exact(sock, length)
    if body is None:
        raise ProtocolError("Connection closed after a message header")
    return decode(body)


class PolicyClient:
    """Persistent connection to a policy server (one client per server: the server keeps the action queue)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        timeout: float = 600.0,
        connect_timeout: float = 600.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        deadline = time.monotonic() + self.connect_timeout
        while True:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def call(self, cmd: str, obs: Any = None) -> Any:
        self.connect()
        try:
            send_message(self._sock, {"cmd": cmd, "obs": obs})
            response = recv_message(self._sock)
        except BaseException:
            self.close()
            raise
        if response is None:
            self.close()
            raise ProtocolError("The server closed the connection")
        if "error" in response:
            self.close()
            raise ServerError(f"{response['error']}\n{response.get('traceback', '')}")
        return response.get("res")

    def reset(self) -> None:
        self.call("reset_model")

    def predict(self, obs: dict) -> np.ndarray:
        return np.asarray(self.call("predict", obs))

    def __enter__(self) -> PolicyClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

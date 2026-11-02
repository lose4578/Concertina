import contextlib
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from concertina.eval import server as server_module
from concertina.eval.server import (
    COMMANDS,
    EvalPolicy,
    PolicyServer,
    ServerConfig,
    load_policy,
    local_policy_dir,
    resolve_policy_path,
)
from concertina_eval_client.protocol import PolicyClient, ServerError, recv_message, send_message


class RecordingModel:
    """Stands in for `EvalPolicy`; `secret` must never be reachable over the wire."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def reset_model(self) -> None:
        self.calls.append(("reset_model", None))

    def predict(self, obs):
        self.calls.append(("predict", obs))
        if obs.get("fail"):
            raise ValueError("bad observation")
        return np.full(7, float(len(self.calls)), dtype=np.float32)

    def secret(self, *args):
        raise AssertionError("dispatched a non-whitelisted method")


@pytest.fixture
def server():
    model = RecordingModel()
    server = PolicyServer(model, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, model
    server.close()


def _raw_call(port: int, request) -> dict | None:
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        send_message(sock, request)
        response = recv_message(sock)
        # After an error the server closes the connection.
        return response, recv_message(sock) if "error" in (response or {}) else "open"


def test_whitelisted_commands_round_trip(server):
    server, model = server
    with PolicyClient(port=server.address[1], timeout=10, connect_timeout=10) as client:
        client.reset()
        action = client.predict({"images": {}, "instruction": "go"})
    assert action.dtype == np.float32 and action.shape == (7,)
    assert [name for name, _ in model.calls] == ["reset_model", "predict"]
    assert model.calls[1][1]["instruction"] == "go"


@pytest.mark.parametrize("cmd", ["secret", "__init__", "__class__", "model", "dispatch", "", None, 3])
def test_other_commands_are_rejected_without_dispatch(server, cmd):
    server, model = server
    response, after = _raw_call(server.address[1], {"cmd": cmd, "obs": None})
    assert "Unknown command" in response["error"]
    assert after is None
    assert model.calls == []


def test_non_object_request_is_rejected(server):
    server, _ = server
    response, _ = _raw_call(server.address[1], ["predict"])
    assert "JSON object" in response["error"]


def test_server_survives_failures_and_garbage(server):
    server, model = server
    port = server.address[1]
    with (
        PolicyClient(port=port, timeout=10, connect_timeout=10) as client,
        pytest.raises(ServerError, match="bad observation"),
    ):
        client.predict({"fail": True})
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(b"GET / HTTP/1.1\r\n\r\n")
        # The server drops the connection (a reset when the request was not fully read).
        with contextlib.suppress(ConnectionResetError):
            assert sock.recv(1) == b""
    with PolicyClient(port=port, timeout=10, connect_timeout=10) as client:
        assert client.predict({}).shape == (7,)


def test_defaults_bind_loopback_and_whitelist():
    assert ServerConfig().host == "127.0.0.1"
    assert COMMANDS == ("predict", "reset_model")


class FakePolicy:
    def __init__(self) -> None:
        self.queue = 0
        self.batches = []

    def count_queued_actions(self) -> int:
        return self.queue

    def select_action(self, batch):
        self.batches.append(batch)
        self.queue = 1 if self.queue == 0 else 0
        return torch.arange(20, dtype=torch.float32).unsqueeze(0)

    def reset(self) -> None:
        self.queue = 0


class Identity:
    def __call__(self, value):
        return value

    def reset(self) -> None:
        pass


def test_eval_policy_maps_environment_observations():
    policy = FakePolicy()
    model = EvalPolicy(
        policy,
        Identity(),
        lambda action: action[:, :7],
        image_keys={"rgb_static": "observation.images.top"},
        state_keys={"robot_obs": "observation.state"},
    )
    image = np.random.default_rng(0).integers(0, 255, (5, 6, 3), dtype=np.uint8)
    obs = {
        "images": {"rgb_static": image},
        "state": {"robot_obs": np.arange(15)},
        "instruction": "open drawer",
    }
    for _ in range(3):
        action = model.predict(obs)
    np.testing.assert_array_equal(action, np.arange(7, dtype=np.float32))
    assert model.num_decodes == 2
    batch = policy.batches[0]
    assert batch["task"] == ["open drawer"]
    assert batch["observation.images.top"].dtype == torch.uint8
    assert torch.equal(batch["observation.images.top"][0], torch.from_numpy(image).permute(2, 0, 1))
    assert batch["observation.state"].dtype == torch.float32 and batch["observation.state"].shape == (1, 15)


@pytest.mark.parametrize(
    "obs",
    [
        {"images": {}, "state": {"robot_obs": np.zeros(15)}},
        {"images": {"rgb_static": np.zeros((4, 4, 3), np.float32)}, "state": {"robot_obs": np.zeros(15)}},
        {"images": {"rgb_static": np.zeros((4, 4, 4), np.uint8)}, "state": {"robot_obs": np.zeros(15)}},
        {"images": {"rgb_static": np.zeros((4, 4, 3), np.uint8)}, "state": {}},
    ],
)
def test_eval_policy_rejects_incomplete_observations(obs):
    model = EvalPolicy(
        FakePolicy(),
        Identity(),
        Identity(),
        {"rgb_static": "observation.images.top"},
        {"robot_obs": "observation.state"},
    )
    with pytest.raises((KeyError, ValueError)):
        model.predict(obs)


def test_silent_connection_does_not_block_the_client_slot():
    model = RecordingModel()
    server = PolicyServer(model, "127.0.0.1", 0, first_request_timeout_s=0.5)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with (
            socket.create_connection(("127.0.0.1", server.address[1]), timeout=10),
            PolicyClient(port=server.address[1], timeout=10, connect_timeout=10) as client,
        ):
            assert client.predict({}).shape == (7,)
    finally:
        server.close()


def test_policy_paths_resolve_locally_or_pass_hub_ids_through(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "policy").mkdir()
    assert resolve_policy_path("policy") == str(tmp_path / "policy")
    assert resolve_policy_path("org/fake-policy") == "org/fake-policy"
    assert local_policy_dir("policy") == Path("policy")


def test_hub_policy_is_loaded_from_its_snapshot(tmp_path, monkeypatch):
    """Config, weights and processors all come from one snapshot of the repository (no network here)."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    downloads, loaded = [], {}
    monkeypatch.setattr(
        server_module, "snapshot_download", lambda repo_id: downloads.append(repo_id) or str(snapshot)
    )

    def load_config(path):
        loaded["config"] = path
        return SimpleNamespace(type="concertina", device=None, model=SimpleNamespace(inference="saved"))

    def load_weights(path, config):
        loaded["weights"] = path
        return SimpleNamespace(eval=lambda: None, config=config)

    def load_processors(config, pretrained_path, preprocessor_overrides):
        loaded["processors"] = pretrained_path
        loaded["device"] = preprocessor_overrides["device_processor"]["device"]
        return "pre", "post"

    monkeypatch.setattr(server_module.PreTrainedConfig, "from_pretrained", load_config)
    monkeypatch.setattr(server_module.ConcertinaPolicy, "from_pretrained", load_weights)
    monkeypatch.setattr(server_module, "make_pre_post_processors", load_processors)
    policy, pre, post = load_policy("org/fake-policy", "cuda:3", None)
    assert downloads == ["org/fake-policy"]
    assert loaded == {"config": snapshot, "weights": snapshot, "processors": snapshot, "device": "cuda:3"}
    assert policy.config.device == "cuda:3" and policy.config.model.inference == "saved"
    assert (pre, post) == ("pre", "post")

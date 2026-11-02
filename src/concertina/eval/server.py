"""Policy server for simulation clients that run in their own environment (CALVIN).

`concertina-eval-server` loads a Concertina policy directory with its saved pre/post-processors and serves it over the
length-prefixed JSON protocol of `concertina_eval_client.protocol`. Only the commands in `COMMANDS` are dispatched.

Observations are benchmark-neutral::

    {"images": {<env camera>: uint8 [H, W, 3]}, "state": {<env state>: float [D]}, "instruction": str}

`image_keys` / `state_keys` rename them to the dataset keys the checkpoint was trained on (e.g. CALVIN
`rgb_static` -> `observation.images.top`); the checkpoint's preprocessor then converts them to the canonical
layout. `predict` returns one native environment action per call, decoding a new chunk whenever the policy's
action queue is empty (the LeRobot `select_action` contract). The queue is per server, so a server serves one
client at a time and the client calls `reset_model` at every episode (CALVIN: every subtask) start.
"""

import logging
import socket
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

from concertina.policy.config_model import ConcertinaInferenceConfig
from concertina.policy.modeling_concertina import ConcertinaPolicy
from concertina_eval_client.protocol import (
    DEFAULT_MAX_MESSAGE_BYTES,
    ProtocolError,
    recv_message,
    send_message,
)

logger = logging.getLogger(__name__)

COMMANDS = ("predict", "reset_model")


@dataclass
class ServerConfig:
    # Policy directory (config.json, model.safetensors, policy_{pre,post}processor*.json) or Hub repository id.
    policy_path: str = ""
    # Loopback by default: the protocol has no authentication.
    host: str = "127.0.0.1"
    port: int = 9999
    device: str = "cuda"
    # Decoding recipe replacing the checkpoint's saved one; None keeps the checkpoint recipe.
    inference: ConcertinaInferenceConfig | None = None
    # Environment camera / state names -> dataset keys of the checkpoint's inference dataset.
    image_keys: dict[str, str] = field(default_factory=dict)
    state_keys: dict[str, str] = field(default_factory=dict)
    # TF32 matmuls (torch default off); the CALVIN reference runs left it off.
    allow_tf32: bool = False
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    # A connection must send its first request within this many seconds; port scanners and stalled peers would
    # otherwise block the single client slot.
    first_request_timeout_s: float = 60.0


class EvalPolicy:
    """A Concertina policy with its processors, called on raw environment observations."""

    def __init__(
        self,
        policy: ConcertinaPolicy,
        preprocessor: Any,
        postprocessor: Any,
        image_keys: dict[str, str],
        state_keys: dict[str, str],
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.image_keys = dict(image_keys)
        self.state_keys = dict(state_keys)
        self.num_decodes = 0

    def reset_model(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _batch(self, obs: dict) -> dict[str, Any]:
        if not isinstance(obs, dict):
            raise TypeError(f"predict expects an observation mapping, got {type(obs).__name__}")
        images, state = obs.get("images") or {}, obs.get("state") or {}
        batch: dict[str, Any] = {}
        for source, key in self.image_keys.items():
            if source not in images:
                raise KeyError(f"Observation lacks camera {source!r}; got {sorted(images)}")
            image = np.asarray(images[source])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"Camera {source!r} must be uint8 [H, W, 3], got {image.dtype} {image.shape}"
                )
            # uint8 frames are scaled to [0, 1] by the preprocessor, as in training.
            batch[key] = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        for source, key in self.state_keys.items():
            if source not in state:
                raise KeyError(f"Observation lacks state {source!r}; got {sorted(state)}")
            batch[key] = torch.from_numpy(np.asarray(state[source], dtype=np.float32).reshape(1, -1).copy())
        batch["task"] = [str(obs.get("instruction", ""))]
        return batch

    def predict(self, obs: dict) -> np.ndarray:
        queued = self.policy.count_queued_actions()
        batch = self.preprocessor(self._batch(obs))
        action = self.postprocessor(self.policy.select_action(batch))
        if queued == 0:
            self.num_decodes += 1
        return action[0].detach().cpu().numpy().astype(np.float32)


def resolve_policy_path(policy_path: str) -> str:
    """The absolute path of a local policy directory; anything else (a Hub repository id) is returned unchanged."""
    path = Path(policy_path).expanduser()
    return str(path.resolve()) if path.exists() else policy_path


def local_policy_dir(policy_path: str) -> Path:
    """A local policy directory, or the snapshot of a Hub repository id in the Hugging Face cache."""
    path = Path(policy_path).expanduser()
    return path if path.is_dir() else Path(snapshot_download(policy_path))


def load_policy(
    policy_path: str, device: str, inference: ConcertinaInferenceConfig | None
) -> tuple[ConcertinaPolicy, Any, Any]:
    """A Concertina policy (a local directory or a Hub repository id) with its saved pre/post-processors.

    ``inference`` replaces the decoding recipe saved with the checkpoint (None keeps it).
    """
    path = local_policy_dir(policy_path)
    policy_config = PreTrainedConfig.from_pretrained(path)
    if policy_config.type != "concertina":
        raise ValueError(f"{policy_path} holds a {policy_config.type!r} policy, not a Concertina policy")
    if inference is not None:
        policy_config.model.inference = inference
    policy_config.device = device
    policy = ConcertinaPolicy.from_pretrained(path, config=policy_config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    logger.info("Loaded %s | inference=%s", policy_path, policy_config.model.inference)
    return policy, preprocessor, postprocessor


def load_eval_policy(config: ServerConfig) -> EvalPolicy:
    if not config.image_keys:
        raise ValueError("image_keys must map at least one environment camera")
    policy, preprocessor, postprocessor = load_policy(config.policy_path, config.device, config.inference)
    return EvalPolicy(policy, preprocessor, postprocessor, config.image_keys, config.state_keys)


class PolicyServer:
    """Serves one client connection at a time and dispatches only `COMMANDS`."""

    def __init__(
        self,
        model: Any,
        host: str = "127.0.0.1",
        port: int = 9999,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        first_request_timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.max_message_bytes = max_message_bytes
        self.first_request_timeout_s = first_request_timeout_s
        self._socket = socket.create_server((host, port))
        self.address = self._socket.getsockname()[:2]
        self._closed = False

    def close(self) -> None:
        self._closed = True
        self._socket.close()

    def serve_forever(self) -> None:
        logger.info("Policy server listening on %s:%d", *self.address)
        while True:
            try:
                connection, peer = self._socket.accept()
            except OSError:
                if self._closed:
                    return
                raise
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                logger.info("Client connected from %s:%d", *peer[:2])
                self.serve_connection(connection)
                logger.info("Client disconnected")

    def serve_connection(self, connection: socket.socket) -> None:
        connection.settimeout(self.first_request_timeout_s)
        while True:
            try:
                request = recv_message(connection, self.max_message_bytes)
            except (ProtocolError, OSError) as exc:
                logger.warning("Dropping connection: %s", exc)
                return
            if request is None:
                return
            connection.settimeout(None)
            try:
                response = {"res": self.dispatch(request)}
            except Exception as exc:
                logger.exception("Request failed")
                # The action queue may be inconsistent after a failure, so the connection is closed.
                self._reply(
                    connection, {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                )
                return
            if not self._reply(connection, response):
                return

    def dispatch(self, request: Any) -> Any:
        if not isinstance(request, dict):
            raise ProtocolError("A request must be a JSON object")
        cmd = request.get("cmd")
        if cmd not in COMMANDS:
            raise ProtocolError(f"Unknown command {cmd!r}; allowed: {list(COMMANDS)}")
        if cmd == "reset_model":
            return self.model.reset_model()
        return self.model.predict(request.get("obs"))

    @staticmethod
    def _reply(connection: socket.socket, payload: dict) -> bool:
        try:
            send_message(connection, payload)
            return True
        except OSError as exc:
            logger.warning("Could not reply: %s", exc)
            return False


@draccus.wrap()
def main(config: ServerConfig) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    start = time.perf_counter()
    model = load_eval_policy(config)
    server = PolicyServer(
        model, config.host, config.port, config.max_message_bytes, config.first_request_timeout_s
    )
    logger.info("Ready after %.1fs", time.perf_counter() - start)
    try:
        server.serve_forever()
    finally:
        server.close()


if __name__ == "__main__":
    main()

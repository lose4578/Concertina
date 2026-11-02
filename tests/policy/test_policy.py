import json
import shutil
from pathlib import Path

import huggingface_hub.constants
import pytest
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from safetensors.torch import load_file, save_file

from concertina.model.rtc import RTCActionQueue, stack_rtc_prefix
from concertina.policy.config_model import ConcertinaActionHistoryConfig, ConcertinaInferenceConfig
from concertina.policy.configuration_concertina import ConcertinaConfig
from concertina.policy.modeling_concertina import ConcertinaPolicy


@pytest.fixture
def policy(tiny_config) -> ConcertinaPolicy:
    torch.manual_seed(0)
    return ConcertinaPolicy(ConcertinaConfig(model=tiny_config, device="cpu"))


def test_factory_resolves_concertina(policy):
    assert get_policy_class("concertina") is ConcertinaPolicy
    assert policy.name == "concertina" and ConcertinaPolicy.config_class is ConcertinaConfig


def test_forward_returns_loss_and_scalar_metrics(policy, make_batch):
    policy.train()
    loss, output = policy(make_batch(20, training=True))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(isinstance(value, float) for value in output.values())
    assert {"loss_action_reconstruction", "loss_token_risk"} <= set(output)
    assert policy.model.training_step.item() == 1


def test_select_action_uses_a_queue_of_n_action_steps(policy, make_batch, tiny_config):
    tiny_config.inference.n_action_steps = 10
    policy.reset()
    batch = make_batch(20, training=False)
    first = policy.select_action(batch)
    assert first.shape == (2, 20) and policy.count_queued_actions() == 9
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (2, 20, 20)
    assert torch.equal(first, chunk[:, 0])
    for step in range(1, 10):
        assert torch.equal(policy.select_action(batch), chunk[:, step])
    assert policy.count_queued_actions() == 0
    policy.select_action(batch)
    policy.reset()
    assert policy.count_queued_actions() == 0
    with pytest.raises(ValueError, match="noise"):
        policy.predict_action_chunk(batch, noise=torch.zeros(1))


def test_optim_params_cover_trainable_parameters(policy, tiny_config):
    groups = {group["name"]: group["params"] for group in policy.get_optim_params()}
    assert set(groups) == {"backbone", "action_heads", "other"}
    assert sum(len(params) for params in groups.values()) == sum(
        1 for parameter in policy.parameters() if parameter.requires_grad
    )
    tiny_config.freeze_action_heads = True
    frozen = ConcertinaPolicy(ConcertinaConfig(model=tiny_config, device="cpu"))
    assert "action_heads" not in {group["name"] for group in frozen.get_optim_params()}


def test_fsdp_wrap_units_exist(policy):
    module_classes = {type(module).__name__ for module in policy.modules()}
    assert set(ConcertinaPolicy._fsdp_wrap_modules) <= module_classes


def test_save_and_load_roundtrip(policy, make_batch, tmp_path):
    policy.train()
    policy(make_batch(20, training=True))
    policy.eval()
    batch = make_batch(20, training=False)
    expected = policy.predict_action_chunk(batch)
    policy.save_pretrained(tmp_path)
    assert (tmp_path / "config.json").exists() and (tmp_path / "model.safetensors").exists()
    assert json.loads((tmp_path / "config.json").read_text())["type"] == "concertina"

    loaded = ConcertinaPolicy.from_pretrained(tmp_path)
    assert isinstance(loaded.config, ConcertinaConfig)
    assert loaded.model.training_step.item() == 1
    assert torch.equal(loaded.predict_action_chunk(batch), expected)
    for name, tensor in policy.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name


def test_loading_is_strict(policy, tmp_path):
    policy.save_pretrained(tmp_path)
    weights = load_file(str(tmp_path / "model.safetensors"))
    weights.pop("model.action_head.0.bias")
    save_file(weights, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="action_head.0.bias"):
        ConcertinaPolicy.from_pretrained(tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    config["model"]["unknown_option"] = 1
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(Exception, match="unknown_option"):
        PreTrainedConfig.from_pretrained(tmp_path)


def test_select_action_records_action_history(tiny_config, make_batch):
    tiny_config.action_history = ConcertinaActionHistoryConfig(enabled=True, horizon=10)
    tiny_config.inference.n_action_steps = 4
    torch.manual_seed(0)
    policy = ConcertinaPolicy(ConcertinaConfig(model=tiny_config, device="cpu"))
    batch = make_batch(20, training=False)
    executed = [policy.select_action(batch) for _ in range(6)]
    assert len(policy._action_history) == 6
    assert all(torch.equal(a, b) for a, b in zip(policy._action_history, executed, strict=True))
    # The second chunk (actions 4 and 5) was decoded with the four executed actions as history.
    history = torch.zeros(2, 10, 20, dtype=torch.bfloat16)
    history[:, -4:] = torch.stack(executed[:4], dim=1).bfloat16()
    valid = torch.zeros(2, 10, dtype=torch.bool)
    valid[:, -4:] = True
    expected = policy.model.predict_action_chunk(batch, 4, history=history, history_valid=valid)
    assert torch.equal(torch.stack(executed[4:], dim=1), expected[:, :2])
    policy.reset()
    assert len(policy._action_history) == 0


@pytest.mark.parametrize("source", ["executed_tail", "leftover_tail"])
def test_select_action_with_real_time_chunking(tiny_config, make_batch, source):
    tiny_config.inference = ConcertinaInferenceConfig(
        n_action_steps=10, rtc=True, rtc_prefix_source=source, rtc_execution_horizon=5
    )
    torch.manual_seed(0)
    policy = ConcertinaPolicy(ConcertinaConfig(model=tiny_config, device="cpu"))
    batch = make_batch(20, training=False)
    first_chunk = policy.model.predict_action_chunk(batch)
    first = [policy.select_action(batch, inference_delay=1) for _ in range(4)]
    # The delay drops the first action; the execution horizon leaves 5 - 1 actions.
    assert all(torch.equal(action, first_chunk[:, 1 + i]) for i, action in enumerate(first))
    assert policy.count_queued_actions() == 0
    second = policy.select_action(batch)
    prefix = first_chunk[:, 1:5] if source == "executed_tail" else first_chunk[:, 5:]
    second_chunk = policy.model.predict_action_chunk(batch, prefix_actions=prefix)
    skipped = prefix.shape[1] if source == "executed_tail" else 0
    assert torch.equal(second, second_chunk[:, skipped])
    assert policy.count_queued_actions() == 4
    policy.drop_queued_actions()
    assert policy.count_queued_actions() == 0


def test_rtc_action_queue():
    queue = RTCActionQueue(execution_horizon=4)
    assert queue.empty() and queue.leftover() is None and queue.executed_tail(4) is None
    chunk = torch.arange(10, dtype=torch.float32).unsqueeze(-1)
    queue.replace(chunk, delay=1, prefix_length=2)
    # Skip the delayed action and the 2-action prefix; 4 - 1 actions are executable.
    assert len(queue) == 3 and [queue.get().item() for _ in range(2)] == [3.0, 4.0]
    assert queue.executed_tail(4).flatten().tolist() == [3.0, 4.0]
    assert queue.leftover().flatten().tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]
    queue.drop()
    assert queue.empty() and queue.executed_tail(1).flatten().tolist() == [4.0]
    rows = [RTCActionQueue(2), RTCActionQueue(2)]
    for row in rows:
        row.replace(chunk, delay=0, prefix_length=0)
        row.get()
    assert stack_rtc_prefix(rows, "executed_tail", chunk_size=10).shape == (2, 1, 1)
    assert (
        stack_rtc_prefix(rows, "leftover_tail", chunk_size=4).flatten().tolist() == [1.0, 2.0, 3.0, 4.0] * 2
    )


def test_a_policy_built_for_a_checkpoint_reads_no_backbone_weights(tiny_config, tmp_path):
    """Resuming from a sharded checkpoint constructs the policy directly; its weights come from the checkpoint."""
    backbone = tmp_path / "backbone"
    shutil.copytree(tiny_config.backbone_path, backbone, ignore=shutil.ignore_patterns("*.safetensors*"))
    tiny_config.backbone_path = ""  # Resolves to the bundled `backbone/` directory of the checkpoint.
    policy = ConcertinaPolicy(ConcertinaConfig(model=tiny_config, device="cpu", pretrained_path=tmp_path))
    assert policy.config.model.backbone_path == str(backbone)
    with pytest.raises(OSError):
        ConcertinaPolicy(ConcertinaConfig(model=policy.config.model, device="cpu"))
    tiny_config.backbone_path = ""
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="bundles no"):
        ConcertinaPolicy(
            ConcertinaConfig(model=tiny_config, device="cpu", pretrained_path=tmp_path / "empty")
        )
    with pytest.raises(FileNotFoundError, match="neither a local directory"):
        ConcertinaPolicy(
            ConcertinaConfig(model=tiny_config, device="cpu", pretrained_path=tmp_path / "missing")
        )


def _cache_as_hub_snapshot(cache: Path, repo_id: str, policy_dir: Path, commit: str, ref: str) -> Path:
    """Lay out `policy_dir` as the cached Hub snapshot `commit` of `repo_id`, named by the branch or tag `ref`."""
    repo = cache / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / commit
    shutil.copytree(policy_dir, snapshot)
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs" / ref).write_text(commit)
    return snapshot


def test_a_hub_repository_id_resolves_the_bundled_backbone_of_its_snapshot(
    policy, make_batch, tiny_config, tmp_path, monkeypatch
):
    """A checkpoint named by Hub id (e.g. `policy.pretrained_path` of a fine-tuning run) loads from the Hugging Face
    cache, with its bundled `backbone/` taken from the snapshot of the requested revision."""
    policy.eval()
    batch = make_batch(20, training=False)
    expected = policy.predict_action_chunk(batch)
    release = tmp_path / "release"
    policy.save_pretrained(release)
    config = json.loads((release / "config.json").read_text())
    config["model"]["backbone_path"] = ""
    (release / "config.json").write_text(json.dumps(config))
    shutil.copytree(
        tiny_config.backbone_path, release / "backbone", ignore=shutil.ignore_patterns("*.safetensors*")
    )
    cache = tmp_path / "hub"
    old = _cache_as_hub_snapshot(cache, "concertina-test/tiny", release, "a" * 40, ref="v1")
    new = _cache_as_hub_snapshot(cache, "concertina-test/tiny", release, "b" * 40, ref="main")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)

    loaded = ConcertinaPolicy.from_pretrained("concertina-test/tiny")
    assert loaded.config.model.backbone_path == str(new / "backbone")
    assert torch.equal(loaded.predict_action_chunk(batch), expected)
    pinned = ConcertinaPolicy.from_pretrained("concertina-test/tiny", revision="v1")
    assert pinned.config.model.backbone_path == str(old / "backbone")

    # Direct construction (resuming from a sharded checkpoint) honors `pretrained_revision`.
    tiny_config.backbone_path = ""
    built = ConcertinaPolicy(
        ConcertinaConfig(
            model=tiny_config, device="cpu", pretrained_path="concertina-test/tiny", pretrained_revision="v1"
        )
    )
    assert built.config.model.backbone_path == str(old / "backbone")


def _changed_fraction(dtype: torch.dtype | None, tiny_config, make_batch) -> float:
    """Fraction of trainable weights with a non-zero gradient that one AdamW step at lr 2.5e-5 changes."""
    torch.manual_seed(0)
    policy = ConcertinaPolicy(ConcertinaConfig(model=tiny_config, device="cpu", dtype=dtype))
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=2.5e-5, weight_decay=0.0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = policy(make_batch(20, training=True))
    loss.backward()
    graded = [p.grad != 0 for p in params]
    optimizer.step()
    changed = sum(
        ((p.detach() != b) & g).sum().item() for p, b, g in zip(params, before, graded, strict=True)
    )
    return changed / sum(g.sum().item() for g in graded)


def test_float32_storage_keeps_small_updates_under_bf16_autocast(tiny_config, make_batch):
    """Training stores fp32 weights (`dtype: float32`) while the loop's bf16 autocast computes; bf16 storage would
    round most lr-2.5e-5 updates away."""
    assert _changed_fraction(torch.float32, tiny_config, make_batch) > 0.99
    assert _changed_fraction(None, tiny_config, make_batch) < 0.5


def test_float32_storage_decodes_like_bfloat16(tiny_config, make_batch, tmp_path):
    torch.manual_seed(0)
    reference = ConcertinaPolicy(ConcertinaConfig(model=tiny_config, device="cpu"))
    torch.manual_seed(0)
    policy = ConcertinaPolicy(ConcertinaConfig(model=tiny_config, device="cpu", dtype=torch.float32))
    assert {p.dtype for p in policy.parameters()} == {torch.float32}
    batch = make_batch(20, training=False)
    expected = reference.predict_action_chunk(batch)
    assert torch.equal(policy.predict_action_chunk(batch), expected)
    assert {p.dtype for p in policy.parameters()} == {torch.float32}
    policy.reset()
    assert torch.equal(policy.select_action(batch), expected[:, 0])

    policy.save_pretrained(tmp_path)
    loaded = ConcertinaPolicy.from_pretrained(tmp_path)
    assert loaded.config.dtype == torch.float32 and loaded.model.action_head[0].weight.dtype == torch.float32
    assert torch.equal(loaded.predict_action_chunk(batch), expected)

"""Kept implementation branches: block topology, corruption, visible action context, decoding strategies, real-time
chunking and action history (bitwise parity with the original implementation is checked by the parity matrix)."""

import dataclasses

import pytest
import torch

from concertina.model import ConcertinaModel
from concertina.model.attention import build_history_block_attention_mask, build_training_attention_mask
from concertina.model.diffusion import corrupt_action_tokens, mask_values, perturb_visible_context
from concertina.model.inputs import mask_state_dims
from concertina.model.model import split_history_window
from concertina.model.rng import SampleKeyedRNG
from concertina.model.sampler import draft_ranking, refine_block, select_commits, token_confidence
from concertina.policy.config_model import (
    ConcertinaActionHistoryConfig,
    ConcertinaCorruptionConfig,
    ConcertinaInferenceConfig,
    ConcertinaModelConfig,
    ConcertinaVisibleContextConfig,
)

RNG = SampleKeyedRNG.from_values([3, 11], dataset_ids=[0, 1], seed=5, step=7)


def _model(config: ConcertinaModelConfig, **changes) -> ConcertinaModel:
    torch.manual_seed(0)
    return ConcertinaModel(dataclasses.replace(config, **changes))


def _history_batch(batch: dict, horizon: int, pad: int) -> dict:
    """Prepend `horizon` history steps (the first `pad` of sample 0 padded) to a training batch."""
    batch = dict(batch)
    generator = torch.Generator().manual_seed(9)
    history = torch.randn(2, horizon, 20, generator=generator)
    gripper = torch.rand(2, horizon, 2, generator=generator)
    history[..., [9, 19]] = gripper
    is_pad = torch.zeros(2, horizon, dtype=torch.bool)
    is_pad[0, :pad] = True
    batch["action"] = torch.cat([history, batch["action"]], dim=1)
    batch["action_gripper_binary"] = torch.cat([gripper, batch["action_gripper_binary"]], dim=1)
    batch["action_is_pad"] = torch.cat([is_pad, batch["action_is_pad"]], dim=1)
    return batch


# Configuration.


def test_config_validation():
    with pytest.raises(ValueError, match="multiple of block_size"):
        ConcertinaModelConfig(chunk_size=20, block_size=8)
    with pytest.raises(ValueError, match="token_risk"):
        ConcertinaModelConfig(token_risk=False)
    with pytest.raises(ValueError, match="log_linear_mask"):
        ConcertinaModelConfig(
            full_mask_probability=0.2, corruption=ConcertinaCorruptionConfig(mode="bard_mixed")
        )
    with pytest.raises(ValueError, match="multiple of block_size"):
        ConcertinaModelConfig(
            action_history=ConcertinaActionHistoryConfig(enabled=True, horizon=15, mode="native")
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        ConcertinaModelConfig(
            action_history=ConcertinaActionHistoryConfig(enabled=True),
            inference=ConcertinaInferenceConfig(rtc=True),
        )
    with pytest.raises(ValueError, match="positive noise_std"):
        ConcertinaVisibleContextConfig(strategy="shared_gaussian", apply_probability=0.5)
    with pytest.raises(ValueError, match="sequential"):
        ConcertinaInferenceConfig(commit_strategy="sequential", draft_ranking="random")
    nested = ConcertinaModelConfig(corruption={"mode": "bard_mixed"}, action_history={"enabled": True})
    assert nested.corruption.mode == "bard_mixed" and nested.action_history.enabled


# Training-side primitives.


def test_bard_mixed_corruption():
    tokens = torch.randn(2, 30, 4)
    timesteps = torch.rand(2, 30).clamp_min(0.05)
    valid = torch.ones(2, 30, dtype=torch.bool)
    valid[1, -3:] = False
    masked = torch.full_like(tokens, 7.0)
    corruption = ConcertinaCorruptionConfig(mode="bard_mixed", visible_uniform_range=(-0.5, 0.5))
    noisy, supervised, weights, visible = corrupt_action_tokens(
        tokens, timesteps, valid, RNG, masked_values=masked, corruption=corruption, eps=1e-3
    )
    is_masked = noisy.eq(7.0).all(-1) & valid
    clean = noisy.eq(tokens).all(-1)
    assert torch.equal(supervised, (is_masked | visible) & valid) and not (clean & supervised).any()
    shown = visible & valid  # Padded tokens keep their clean values.
    assert ((noisy[shown] >= -0.5) & (noisy[shown] <= 0.5)).all() and shown.any() and is_masked.any()
    assert torch.equal(weights, torch.ones_like(weights)) and torch.equal(noisy[~valid], tokens[~valid])
    gaussian = dataclasses.replace(corruption, visible_corruption="gaussian", loss_weighting="inverse_t")
    noisy, _, weights, visible = corrupt_action_tokens(
        tokens, timesteps, valid, RNG, masked_values=masked, corruption=gaussian, eps=1e-3
    )
    assert torch.allclose(weights, 1 / timesteps) and not torch.equal(noisy[visible], tokens[visible])
    log_linear = corrupt_action_tokens(
        tokens, timesteps, valid, RNG, masked_values=masked, corruption=ConcertinaCorruptionConfig(), eps=1e-3
    )
    assert not log_linear[3].any()


def test_visible_context_perturbation():
    tokens = torch.zeros(2, 10, 20)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[0, -2:] = False
    dim_mask = torch.ones(2, 20, dtype=torch.bool)
    dim_mask[1, 10:] = False
    config = ConcertinaVisibleContextConfig("block_shared_gaussian", 1.0, [0.1] * 19 + [0.0], 0.0)
    noisy, selected = perturb_visible_context(
        tokens, valid, dim_mask, RNG, config=config, block_size=5, gripper_indices=[9], stream_prefix="t"
    )
    assert selected.all()
    assert noisy[:, :, [9, 19]].eq(0).all() and noisy[1, :, 10:].eq(0).all() and noisy[0, -2:].eq(0).all()
    # One draw per block, shared by its tokens.
    assert torch.equal(noisy[1, 0], noisy[1, 4]) and not torch.equal(noisy[1, 4], noisy[1, 5])


def test_state_mask_ratio_and_fill():
    state = torch.ones(2, 20)
    dims = [0, 1, 2, 3]
    masked = mask_state_dims(state, dims, 1.0, RNG, ratio_range=(0.5, 0.5), fill_value=-2.0)
    assert (masked == -2.0).sum(dim=1).tolist() == [2, 2] and masked[:, 4:].eq(1).all()
    assert torch.equal(
        masked, mask_state_dims(state, dims, 1.0, RNG, ratio_range=(0.5, 0.5), fill_value=-2.0)
    )
    full = mask_state_dims(state, dims, 1.0, RNG, fill_value=3.0)
    assert full[:, dims].eq(3.0).all() and torch.equal(mask_state_dims(state, dims, 0.0, RNG), state)


def test_mask_values():
    assert mask_values("ones", (2, 3), device=torch.device("cpu"), dtype=torch.float32).eq(1).all()
    torch.manual_seed(1)
    first = mask_values("random", (2, 3), device=torch.device("cpu"), dtype=torch.float32)
    torch.manual_seed(1)
    assert torch.equal(first, torch.randn(2, 3))


def test_history_window_split():
    batch = {
        "action": torch.arange(2 * 6 * 1, dtype=torch.float32).view(2, 6, 1),
        "action_is_pad": torch.tensor([[True, False, False, False, False, True]] * 2),
    }
    chunk_batch, history, valid = split_history_window(batch, horizon=2, chunk_size=4)
    assert chunk_batch["action"].shape == (2, 4, 1) and chunk_batch["action_is_pad"][:, -1].all()
    assert valid.tolist() == [[False, True]] * 2 and history[:, 0].eq(0).all()
    future_only = {"action": torch.zeros(2, 4, 1), "action_history_unavailable": torch.tensor(True)}
    _, history, valid = split_history_window(future_only, horizon=2, chunk_size=4)
    assert history.shape == (2, 2, 1) and not valid.any()
    with pytest.raises(ValueError, match="packed window"):
        split_history_window({"action": torch.zeros(2, 4, 1)}, horizon=2, chunk_size=4)


def test_attention_masks_with_native_history():
    # Prefix: 2 context tokens + 4 history tokens (blocks of 2; the first history token is missing).
    prefix_valid = torch.tensor([[True, True, False, True, True, True]])
    action_valid = torch.ones(1, 2, dtype=torch.bool)
    mask = build_training_attention_mask(prefix_valid, action_valid, 2, dtype=torch.float32, history_length=4)
    allowed = (mask[0, 0] == 0).int()
    # Context tokens never see history; history block 1 sees the context and both history blocks.
    assert allowed[1, :6].tolist() == [1, 1, 0, 0, 0, 0]
    assert allowed[4, :6].tolist() == [1, 1, 0, 1, 1, 1]
    assert allowed[3, :6].tolist() == [1, 1, 0, 1, 0, 0]
    # Actions see the valid context and history.
    assert allowed[6, :6].tolist() == [1, 1, 0, 1, 1, 1]
    block = build_history_block_attention_mask(
        prefix_valid[:, :2], prefix_valid[:, 2:4], prefix_valid[:, 4:], dtype=torch.float32
    )
    assert ((block[0, 0] == 0).int()).tolist() == [[1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1]]


# Decoding primitives.


def test_commit_strategies_and_rankings():
    pending = torch.tensor([[True] * 6])
    confidence = torch.tensor([[0.5, 3.0, 2.0, 1.0, -1.0, 0.1]])
    budget = torch.tensor([2])
    dynamic = select_commits(confidence, pending, budget, threshold=0.2, max_ratio=1.0)
    assert dynamic[0].nonzero().flatten().tolist() == [0, 1, 2, 3]
    static = select_commits(confidence, pending, budget, strategy="static", threshold=0.2)
    assert static[0].nonzero().flatten().tolist() == [1, 2]
    sequential = select_commits(confidence, pending, budget, strategy="sequential")
    assert sequential[0].nonzero().flatten().tolist() == [0, 1]
    ranked = select_commits(confidence, pending, budget, strategy="static", ranking=-confidence)
    assert ranked[0].nonzero().flatten().tolist() == [4, 5]
    pending = torch.tensor([[False, True, True, True, True, True]])
    spread = select_commits(confidence, pending, torch.tensor([2]), strategy="static", spread=True)
    # The farthest position from the committed token 0 (5), then the more confident of the two positions
    # farthest from both (2 and 3).
    assert spread[0].nonzero().flatten().tolist() == [2, 5]

    hidden, prediction = torch.randn(2, 4, 8), torch.randn(2, 4, 3)
    previous, inputs = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    for name in ("hidden_norm", "prediction_stability", "combined"):
        assert token_confidence(name, hidden, prediction, previous, inputs, None).shape == (2, 4)
    assert torch.equal(
        token_confidence("hidden_norm", hidden, prediction, None, inputs, None), hidden.norm(dim=-1)
    )
    assert token_confidence("prediction_stability", hidden, prediction, None, inputs, None).eq(1).all()
    small = draft_ranking("hidden_norm_small", hidden, prediction, None, inputs, None)
    assert torch.equal(small, -hidden.norm(dim=-1))
    generator = torch.Generator().manual_seed(4)
    first = draft_ranking("random", hidden, prediction, None, inputs, generator)
    assert torch.equal(first, torch.rand(2, 4, generator=torch.Generator().manual_seed(4)))


def test_refine_block_respects_fixed_tokens_and_no_repeat():
    tokens = torch.zeros(1, 4, 2)
    confidence = torch.tensor([[torch.inf, 0.0, 1.0, 2.0]])
    eligible = torch.tensor([[False, True, True, True]])
    selections = []

    def score_fn(draft, previous):
        selections.append(draft[0, :, 0].eq(5.0).tolist())
        return draft, torch.ones_like(draft), torch.full(draft.shape[:2], 10.0)

    recipe = ConcertinaInferenceConfig(
        remask_steps=5,
        remask_ratio=0.34,
        remask_cooldown_steps=0,
        remask_no_repeat=True,
        remask_keep_best=False,
    )
    refined, _ = refine_block(
        tokens, confidence, eligible, score_fn, recipe, lambda: torch.full((1, 4, 2), 5.0)
    )
    # The fixed token is never remasked, every other token exactly once.
    assert selections == [
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
    ]
    assert refined[0, 0].eq(0).all() and refined[0, 1:].eq(1).all()


# Model.


@pytest.mark.parametrize(
    "changes",
    [
        {"chunk_size": 20, "block_size": 5},
        {"corruption": ConcertinaCorruptionConfig(mode="bard_mixed", loss_weighting="inverse_t")},
        {"corruption": ConcertinaCorruptionConfig(mode="bard_mixed", visible_corruption="gaussian")},
        {"mask_value": "ones", "state_mask_ratio_range": (0.5, 0.5), "state_mask_fill_value": -1.0},
        {"mask_value": "random"},
        {
            "token_risk": False,
            "random_resized_crop": False,
            "inference": ConcertinaInferenceConfig(20, confidence="combined"),
        },
    ],
)
def test_training_and_decoding_branches(tiny_config, make_batch, changes):
    model = _model(tiny_config, **changes)
    model.train()
    loss, metrics = model(make_batch(20, training=True))
    loss.backward()
    assert torch.isfinite(loss) and ("loss_token_risk" in metrics) == model.config.token_risk
    assert ("visible_corrupt_ratio" in metrics) == (model.config.corruption.mode == "bard_mixed")
    model.eval()
    actions = model.predict_action_chunk(make_batch(20, training=False))
    assert actions.shape == (2, 20, 20) and torch.isfinite(actions).all()
    if model.token_risk_head is None:
        assert not any(name.startswith("token_risk_head") for name in model.state_dict())


def test_block_size_prefix_decoding(tiny_config, make_batch):
    model = _model(tiny_config, chunk_size=20, block_size=5)
    model.eval()
    batch = make_batch(20, training=False)
    assert torch.equal(
        model.predict_action_chunk(batch, num_actions=5), model.predict_action_chunk(batch)[:, :5]
    )


def test_visible_action_context_training(tiny_config, make_batch):
    context = ConcertinaVisibleContextConfig("shared_gaussian", 1.0, [0.1], repair_loss_weight=0.5)
    model = _model(tiny_config, visible_action_context=context)
    model.train()
    loss, metrics = model(make_batch(20, training=True))
    assert metrics["loss_visible_repair"] > 0
    assert torch.equal(metrics["loss_visible_repair_weighted"], 0.5 * metrics["loss_visible_repair"])
    assert torch.allclose(
        loss.detach(),
        metrics["loss_action_reconstruction"]
        + metrics["loss_visible_repair_weighted"]
        + metrics["loss_token_risk_weighted"],
    )


@pytest.mark.parametrize(
    "inference",
    [
        ConcertinaInferenceConfig(20, confidence="hidden_norm", commit_strategy="static"),
        ConcertinaInferenceConfig(20, confidence="prediction_stability", commit_strategy="sequential"),
        ConcertinaInferenceConfig(20, confidence="combined", draft_ranking="spread_risk"),
        ConcertinaInferenceConfig(20, draft_ranking="random", draft_seed=3),
        ConcertinaInferenceConfig(20, draft_ranking="combined_fp32", remask_steps=4, remask_no_repeat=True),
        ConcertinaInferenceConfig(20, draft_ranking="stability_norm_bootstrap", visible_revision=False),
    ],
)
def test_decoding_strategies_are_deterministic(tiny_config, make_batch, inference):
    model = _model(tiny_config, inference=inference)
    model.eval()
    batch = make_batch(20, training=False)

    def decode(**kwargs):
        model._draft_seed_calls.clear()
        return model.predict_action_chunk(batch, **kwargs)

    first = decode()
    assert torch.equal(first, decode())
    assert torch.equal(decode(num_actions=10), first[:, :10])


def test_random_draft_seed_advances_per_decode(tiny_config, make_batch):
    model = _model(tiny_config, inference=ConcertinaInferenceConfig(20, draft_ranking="random", draft_seed=3))
    model.eval()
    batch = make_batch(20, training=False)
    model.predict_action_chunk(batch)
    model.predict_action_chunk(batch)
    assert model._draft_seed_calls == {3: 2}


def test_keep_draft_translation(tiny_config, make_batch):
    batch = make_batch(20, training=False)
    remask = ConcertinaInferenceConfig(20, remask_steps=3, remask_ratio=0.5, remask_accept_if_improved=False)
    draft_only = _model(tiny_config, inference=dataclasses.replace(remask, remask_steps=0)).eval()
    refined = _model(tiny_config, inference=remask).eval()
    kept = _model(
        tiny_config, inference=dataclasses.replace(remask, remask_keep_draft_translation=True)
    ).eval()
    expected = refined.predict_action_chunk(batch, num_actions=10)
    xyz = [0, 1, 2, 10, 11, 12]
    expected[:, :, xyz] = draft_only.predict_action_chunk(batch, num_actions=10)[:, :, xyz]
    assert torch.equal(kept.predict_action_chunk(batch, num_actions=10), expected)
    with pytest.raises(ValueError, match="10 actions"):
        kept.predict_action_chunk(batch)


def test_rtc_prefix_is_kept(tiny_config, make_batch):
    model = _model(
        tiny_config, inference=ConcertinaInferenceConfig(20, rtc=True, rtc_execution_horizon=5)
    ).eval()
    batch = make_batch(20, training=False)
    prefix = torch.randn(2, 13, 20)
    actions = model.predict_action_chunk(batch, prefix_actions=prefix)
    expected = prefix.masked_fill(~batch["action_valid_dim_mask"][:, None, :], 0).bfloat16().float()
    assert torch.equal(actions[:, :13], expected)
    assert not torch.equal(actions, model.predict_action_chunk(batch))


@pytest.mark.parametrize(
    "history",
    [
        ConcertinaActionHistoryConfig(True, horizon=10, mode="prefix"),
        ConcertinaActionHistoryConfig(True, horizon=10, mode="prefix", missing_history="random_noise"),
        ConcertinaActionHistoryConfig(True, horizon=10, mode="native"),
        ConcertinaActionHistoryConfig(
            True, horizon=20, mode="native", missing_history="random_noise", noise_std=0.5
        ),
    ],
)
def test_action_history(tiny_config, make_batch, history):
    context = ConcertinaVisibleContextConfig("shared_gaussian", 1.0, [0.1], apply_to_history=True)
    model = _model(tiny_config, action_history=history, visible_action_context=context)
    assert model.action_history_role_embedding.weight.shape == (2, model.backbone.hidden_size)
    groups = {group["name"]: group["params"] for group in model.optim_param_groups()}
    assert any(param is model.action_history_role_embedding.weight for param in groups["action_heads"])
    frozen = _model(tiny_config, action_history=history, freeze_action_heads=True)
    assert not frozen.action_history_role_embedding.weight.requires_grad
    model.train()
    batch = make_batch(20, training=True)
    loss, _ = model(_history_batch(batch, history.horizon, pad=3))
    loss.backward()
    assert torch.isfinite(loss) and model.action_history_role_embedding.weight.grad is not None
    loss, _ = model({**batch, "action_history_unavailable": torch.tensor(True)})
    assert torch.isfinite(loss)

    model.eval()
    eval_batch = make_batch(20, training=False)
    values = torch.randn(2, history.horizon, 20)
    valid = torch.zeros(2, history.horizon, dtype=torch.bool)
    valid[:, -4:] = True
    torch.manual_seed(2)
    with_history = model.predict_action_chunk(eval_batch, history=values, history_valid=valid)
    torch.manual_seed(2)
    assert torch.equal(
        with_history, model.predict_action_chunk(eval_batch, history=values, history_valid=valid)
    )
    torch.manual_seed(2)
    assert not torch.equal(with_history, model.predict_action_chunk(eval_batch))

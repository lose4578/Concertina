import pytest
import torch

from concertina.model import ConcertinaModel
from concertina.model.attention import build_block_attention_mask, build_training_attention_mask
from concertina.model.diffusion import per_sample_commit_budgets
from concertina.model.rng import SampleKeyedRNG
from concertina.model.sampler import refine_block, select_commits
from concertina.policy.config_model import ConcertinaInferenceConfig


@pytest.fixture
def model(tiny_config) -> ConcertinaModel:
    torch.manual_seed(0)
    return ConcertinaModel(tiny_config)


def test_training_forward_backward(model, make_batch):
    model.train()
    batch = make_batch(20, training=True)
    loss, metrics = model(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(metrics) >= {"loss_action_reconstruction", "loss_token_risk", "loss_token_risk_weighted"}
    loss.backward()
    assert model.action_head[0].weight.grad is not None
    assert model.token_risk_head[1].weight.grad is not None
    assert model.backbone.language_model.layers[0].mlp.up_proj.weight.grad is not None


def test_training_forward_is_keyed_by_step(model, make_batch):
    model.train()
    model.training_step.fill_(4)
    first, _ = model(make_batch(20, training=True))
    assert model.training_step.item() == 5
    model.training_step.fill_(4)
    repeat, _ = model(make_batch(20, training=True))
    later, _ = model(make_batch(20, training=True))
    assert torch.equal(first, repeat)
    assert not torch.equal(first, later)
    model.eval()
    model(make_batch(20, training=True))
    assert model.training_step.item() == 6 and "training_step" in model.state_dict()


@pytest.mark.parametrize(("gripper_indices", "gripper_target"), [([9, 19], "binary"), ([], "probability")])
def test_gripper_objectives(tiny_config, make_batch, gripper_indices, gripper_target):
    tiny_config.gripper_indices = gripper_indices
    tiny_config.gripper_target = gripper_target
    model = ConcertinaModel(tiny_config)
    model.train()
    batch = make_batch(20, training=True)
    batch["action"][..., [9, 19]] = (batch["action"][..., [9, 19]] > 0.5).float()
    batch["action_gripper_binary"] = batch["action"][..., [9, 19]].clone()
    if not gripper_indices:
        batch.pop("action_gripper_binary")  # Without gripper dimensions the model needs no gripper targets.
    loss, _ = model(batch)
    assert torch.isfinite(loss)
    model.eval()
    actions = model.predict_action_chunk(make_batch(20, training=False))
    assert torch.isfinite(actions).all()


def test_full_mask_probability(tiny_config, make_batch):
    tiny_config.full_mask_probability = 1.0
    model = ConcertinaModel(tiny_config)
    model.train()
    loss, metrics = model(make_batch(20, training=True))
    assert torch.isfinite(loss) and metrics["masked_ratio"].item() == 1.0


def test_optim_param_groups(model):
    groups = {group["name"]: group["params"] for group in model.optim_param_groups()}
    assert len(groups["action_heads"]) == 16 and len(groups["other"]) == 6
    total = sum(len(params) for params in groups.values())
    assert total == sum(1 for p in model.parameters() if p.requires_grad)


def test_sample_keyed_rng_is_independent_of_batch_composition():
    pair = SampleKeyedRNG.from_values([5, 9], dataset_ids=[0, 2], seed=1, step=3)
    single = SampleKeyedRNG.from_values([9], dataset_ids=[2], seed=1, step=3)
    assert torch.equal(
        pair.rand((4,), stream="s", device="cpu")[1], single.rand((4,), stream="s", device="cpu")[0]
    )
    other_step = SampleKeyedRNG.from_values([9], dataset_ids=[2], seed=1, step=4)
    assert not torch.equal(
        single.rand((4,), stream="s", device="cpu"), other_step.rand((4,), stream="s", device="cpu")
    )


def test_predict_action_chunk_shape_and_determinism(model, make_batch):
    model.eval()
    batch = make_batch(20, training=False)
    actions = model.predict_action_chunk(batch)
    assert actions.shape == (2, 20, 20) and actions.dtype == torch.float32
    assert torch.equal(actions, model.predict_action_chunk(batch))
    # Padded dimensions are zero; gripper channels are probabilities.
    assert torch.all(actions[-1, :, 10:] == 0)
    assert torch.all((actions[..., 9] >= 0) & (actions[..., 9] <= 1))


def test_prefix_decoding_matches_full_chunk(model, make_batch):
    model.eval()
    batch = make_batch(20, training=False)
    full = model.predict_action_chunk(batch)
    prefix = model.predict_action_chunk(batch, num_actions=10)
    assert torch.equal(prefix, full[:, :10])


def test_recipes_without_remasking_or_revision(model, make_batch):
    model.eval()
    model.config.inference = ConcertinaInferenceConfig(
        n_action_steps=20, draft_steps=2, visible_revision=False, remask_steps=0
    )
    actions = model.predict_action_chunk(make_batch(20, training=False))
    assert actions.shape == (2, 20, 20) and torch.isfinite(actions).all()


def test_parameter_freeze(tiny_config):
    tiny_config.freeze_action_heads = True
    tiny_config.freeze_backbone_layers = [0]
    model = ConcertinaModel(tiny_config)
    assert not any(p.requires_grad for p in model.action_in_proj.parameters())
    assert not any(p.requires_grad for p in model.action_head.parameters())
    assert all(p.requires_grad for p in model.token_risk_head.parameters())
    assert not any(p.requires_grad for p in model.backbone.language_model.layers[0].parameters())
    assert all(p.requires_grad for p in model.backbone.language_model.layers[1].parameters())


def test_training_attention_mask_topology():
    prefix_valid = torch.tensor([[True, True, False]])
    action_valid = torch.tensor([[True, True, True, False]])
    mask = build_training_attention_mask(prefix_valid, action_valid, block_size=2, dtype=torch.float32)
    allowed = (mask[0, 0] == 0).int()
    prefix, noisy, clean = slice(0, 3), slice(3, 7), slice(7, 11)
    # Prefix: causal over valid keys; the padded query attends only to itself.
    assert allowed[prefix, prefix].tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    # Noisy block 1 sees its own noisy block and clean block 0, never clean block 1.
    assert allowed[3 + 2, noisy].tolist() == [0, 0, 1, 0]
    assert allowed[3 + 2, clean].tolist() == [1, 1, 0, 0]
    # Clean block 1 sees clean blocks 0 and 1 (valid keys only), never noisy tokens.
    assert allowed[7 + 2, clean].tolist() == [1, 1, 1, 0]
    assert allowed[7 + 2, noisy].sum() == 0
    # Every valid action query sees the valid prefix; the padded action queries see only themselves.
    valid_queries = [3, 4, 5, 7, 8, 9]
    assert allowed[valid_queries, :2].all() and allowed[valid_queries, 2].sum() == 0
    assert allowed[6].nonzero().flatten().tolist() == [6] and allowed[10].nonzero().flatten().tolist() == [10]

    block = build_block_attention_mask(prefix_valid, committed_tokens=2, block_tokens=2, dtype=torch.float32)
    assert block.shape == (1, 1, 2, 7)
    assert ((block[0, 0] == 0).int()[0]).tolist() == [1, 1, 0, 1, 1, 1, 1]


def test_commit_budgets_and_selection():
    pending = torch.tensor([[True] * 10, [True] * 7 + [False] * 3])
    budgets = per_sample_commit_budgets(pending, 4)
    assert budgets.tolist() == [[3, 3, 2, 2], [2, 2, 2, 1]]
    confidence = torch.arange(10, dtype=torch.float32).repeat(2, 1) - 5
    selected = select_commits(confidence, pending, budgets[:, 0], threshold=0.0, max_ratio=0.5)
    # Sample 0: four tokens exceed the threshold, more than its budget of three, within the 5-token cap.
    assert selected[0].nonzero().flatten().tolist() == [6, 7, 8, 9]
    # Sample 1: only pending tokens 5 and 6 are confident; the budget of two picks them.
    assert selected[1].nonzero().flatten().tolist() == [5, 6]


def test_refine_block_keeps_best_state():
    tokens = torch.zeros(1, 4, 3)
    confidence = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    eligible = torch.ones(1, 4, dtype=torch.bool)

    def score_fn(draft, previous):
        return draft, torch.ones_like(draft), torch.full(draft.shape[:2], -1.0)

    def masked():
        return torch.zeros(1, 4, 3)

    recipe = ConcertinaInferenceConfig(remask_steps=2, remask_ratio=0.5, remask_accept_if_improved=True)
    refined, refined_confidence = refine_block(tokens, confidence, eligible, score_fn, recipe, masked)
    # Worse proposals are rejected.
    assert torch.equal(refined, tokens) and torch.equal(refined_confidence, confidence)
    recipe = ConcertinaInferenceConfig(
        remask_steps=1, remask_ratio=0.5, remask_accept_if_improved=False, remask_keep_best=False
    )
    refined, refined_confidence = refine_block(tokens, confidence, eligible, score_fn, recipe, masked)
    assert refined[0, :2].eq(1).all() and refined[0, 2:].eq(0).all()


def test_state_with_an_observation_step_dimension(model, make_batch):
    """`observation.state` may be `(B, D)` or carry the observation step, `(B, 1, D)`."""
    model.train()
    batch = make_batch(20, training=True)
    stepped = {**batch, "observation.state": batch["observation.state"].unsqueeze(1)}
    model.training_step.fill_(0)
    loss, _ = model(batch)
    model.training_step.fill_(0)
    stepped_loss, _ = model(stepped)
    assert torch.equal(loss, stepped_loss)
    model.eval()
    batch = make_batch(20, training=False)
    stepped = {**batch, "observation.state": batch["observation.state"].unsqueeze(1)}
    assert torch.equal(model.predict_action_chunk(batch), model.predict_action_chunk(stepped))

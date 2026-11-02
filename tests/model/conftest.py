import os

import pytest
import torch

from concertina.policy.config_model import ConcertinaInferenceConfig, ConcertinaModelConfig

# Tokenizer/image-processor source for the tiny backbone: a local Qwen3-VL directory or a Hub id.
PROCESSOR_SOURCE = os.environ.get("CONCERTINA_TEST_PROCESSOR", "Qwen/Qwen3-VL-2B-Instruct")
CAMERAS = ["observation.images.top", "observation.images.wrist"]


@pytest.fixture(scope="session")
def tiny_backbone(tmp_path_factory) -> str:
    """A randomly initialized two-layer Qwen3-VL with the real processor, saved to disk."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    try:
        processor = Qwen3VLProcessor.from_pretrained(PROCESSOR_SOURCE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen3-VL processor unavailable ({PROCESSOR_SOURCE}): {exc}")
    path = tmp_path_factory.mktemp("tiny_qwen3_vl")
    processor.save_pretrained(path)
    tokenizer = processor.tokenizer
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 151936,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 4096,
            "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
            "tie_word_embeddings": True,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 256,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    Qwen3VLForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(path)
    return str(path)


@pytest.fixture
def tiny_config(tiny_backbone) -> ConcertinaModelConfig:
    return ConcertinaModelConfig(
        backbone_path=tiny_backbone,
        training_attn_implementation="sdpa",
        gradient_checkpointing=False,
        use_liger_kernels=False,
        camera_keys=list(CAMERAS),
        chunk_size=20,
        gripper_indices=[9, 19],
        state_mask_dims=[2, 12],
        inference=ConcertinaInferenceConfig(n_action_steps=20),
    )


def _make_batch(chunk_size: int, *, training: bool, batch_size: int = 2, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    tasks = ["open the drawer", "stack the red block on the blue block"]
    batch: dict = {"task": [tasks[i % len(tasks)] for i in range(batch_size)]}
    batch[CAMERAS[0]] = torch.rand(batch_size, 3, 200, 200, generator=generator)
    batch[CAMERAS[1]] = torch.rand(batch_size, 1, 3, 84, 84, generator=generator)
    batch["observation.state"] = torch.randn(batch_size, 20, generator=generator)
    valid = torch.ones(batch_size, 20, dtype=torch.bool)
    valid[-1, 10:] = False
    batch["action_valid_dim_mask"] = valid
    batch["state_valid_dim_mask"] = valid
    if training:
        action = torch.randn(batch_size, chunk_size, 20, generator=generator)
        gripper = torch.rand(batch_size, chunk_size, 2, generator=generator)
        gripper[~valid[:, 19]] = 0.0  # Padded action dimensions carry a zero gripper target.
        action[..., [9, 19]] = gripper
        is_pad = torch.zeros(batch_size, chunk_size, dtype=torch.bool)
        is_pad[0, -3:] = True
        batch.update(
            {
                "action": action,
                "action_gripper_binary": gripper,
                "action_is_pad": is_pad,
                "index": torch.arange(batch_size) + 100,
                "dataset_index": torch.zeros(batch_size, dtype=torch.long),
            }
        )
    return batch


@pytest.fixture
def make_batch():
    """Factory of synthetic preprocessed batches: `make_batch(chunk_size, training=...)`."""
    return _make_batch

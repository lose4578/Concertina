# Reference environment

The reference environment is a conda env named `concertina` (Python 3.12, linux x86_64) with CUDA 12.8 PyTorch
wheels, which cover Blackwell (`sm_120`) GPUs. The cu128 wheels need NVIDIA driver >= 570.86.

## Install

Run from the repository root. `uv` is required: lerobot declares `[tool.uv.sources]` that route `torch` /
`torchvision` to the PyTorch cu128 index and `torchcodec` to the PyTorch CPU index, and `uv` applies them
when resolving the lerobot git dependency.

```bash
conda create -n concertina python=3.12 pip -y
conda activate concertina
uv pip install --python "$(which python)" torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$(which python)" -c tools/env/constraints.txt -e ".[dev,liger]"
```

Extras: `test` (pytest), `dev` (`test` + ruff), `liger` (liger-kernel fused kernels, Linux + CUDA only).

## Verify

```bash
CUDA_VISIBLE_DEVICES= pytest tests/contract
```

The contract tests check that lerobot discovers the `concertina` policy type through its plugin mechanism and
that the upstream functions `concertina-train` replaces or calls still have the pinned signatures and semantics.

## Pinned versions

| Package | Version |
|---|---|
| python | 3.12.16 |
| torch | 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19; arch list includes `sm_120`) |
| torchvision | 0.26.0+cu128 |
| triton | 3.6.0 |
| lerobot | 0.6.2 @ `2b8aeee410753fcfaa44ec259707474bf6141b62` |
| transformers | 5.5.4 |
| accelerate | 1.15.0 |
| datasets | 4.8.5 |
| torchcodec | 0.11.1+cpu |
| huggingface-hub | 1.33.0 |
| draccus | 0.11.6 |
| numpy | 2.2.6 |
| safetensors | 0.8.0 |
| scipy | 1.18.1 |
| liger-kernel | 0.8.5 (`liger` extra) |

`constraints.txt` holds the complete package set. Regenerate it after any dependency change:

```bash
python -m pip freeze --all --exclude-editable --exclude pip > tools/env/constraints.txt
```

## Packaging note

The distribution name must stay `lerobot_policy_concertina`. lerobot's `register_third_party_plugins` imports
every installed distribution whose metadata `Name` starts with `lerobot_policy_`, using that name as the
module name; `tests/contract/test_plugin_registration.py` fails if a build backend rewrites it.

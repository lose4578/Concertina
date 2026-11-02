"""Constants of the released Concertina model and the batch keys it consumes."""

# Canonical action/state layout: two arms x (xyz, rotation-6d, gripper).
ACTION_DIM = 20
STATE_DIM = ACTION_DIM
# Translation (xyz) dimensions of both arms.
TRANSLATION_DIMS = (0, 1, 2, 10, 11, 12)

# Action/state input projector and action output head (MLPs with SiLU).
ACTION_ENCODER_LAYERS = 4
ACTION_ENCODER_HIDDEN_DIM = 1024
ACTION_HEAD_LAYERS = 4
ACTION_HEAD_HIDDEN_DIM = 1024

# Token-risk head: predicts the log reconstruction loss of each action token; its negation is the
# decoding confidence.
TOKEN_RISK_HIDDEN_DIM = 256
TOKEN_RISK_LOSS_WEIGHT = 0.05
TOKEN_RISK_EPS = 1e-6

# Log-linear masking schedule lower bound.
NOISE_EPS = 1e-3

# Decoding confidences: stability = 1 / (mean absolute change + eps); "combined" weights stability by this factor
# and the hidden-state norm by its complement.
CONFIDENCE_EPS = 1e-6
STABILITY_WEIGHT = 0.5

# Image preprocessing.
IMAGE_SIZE = (224, 224)
# Training-time random resized crop.
CROP_SCALE = (0.9, 1.0)
CROP_RATIO = (1.0, 1.0)

# Bounded LRU size of the tokenized-prompt and MRoPE-position caches.
PROMPT_LAYOUT_CACHE_CAPACITY = 1024

# Batch keys.
OBS_STATE = "observation.state"
ACTION = "action"
ACTION_IS_PAD = "action_is_pad"
TASK = "task"
# Gripper target probabilities in [0, 1], shape (B, T, ACTION_DIM) or (B, T, len(gripper_indices)).
GRIPPER_TARGET = "action_gripper_binary"
# With action history, a true flag marks a batch whose action window holds only the predicted chunk (no history).
ACTION_HISTORY_UNAVAILABLE = "action_history_unavailable"
# Per-sample boolean masks of the valid (non-padded) action/state dimensions, shape (B, D) or (D,).
ACTION_VALID_DIM_MASK = "action_valid_dim_mask"
STATE_VALID_DIM_MASK = "state_valid_dim_mask"
# Stable per-sample keys of the training randomness (see `concertina.model.rng`).
SAMPLE_INDEX = "index"
DATASET_INDEX = "dataset_index"

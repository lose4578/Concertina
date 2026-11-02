from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig

from concertina.policy.config_data import ConcertinaDataConfig
from concertina.policy.config_model import ConcertinaModelConfig


@PreTrainedConfig.register_subclass("concertina")
@dataclass
class ConcertinaConfig(PreTrainedConfig):
    model: ConcertinaModelConfig = field(default_factory=ConcertinaModelConfig)
    data: ConcertinaDataConfig = field(default_factory=ConcertinaDataConfig)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    # Upstream defaults to True, which makes `validate()` require a hub repo_id.
    push_to_hub: bool = False

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        self.data.validate_for_model(
            list(self.model.camera_keys),
            list(self.model.gripper_indices),
            gripper_target=self.model.gripper_target,
        )

    @property
    def chunk_size(self) -> int:
        return self.model.chunk_size

    @property
    def n_action_steps(self) -> int:
        return self.model.inference.n_action_steps

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        history = self.model.action_history
        start = -history.horizon if history.enabled else 0
        return list(range(start, self.model.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def validate_features(self) -> None:
        pass

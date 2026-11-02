"""Batch-level input preparation: camera images, training image augmentation and state dropout."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from concertina.model.constants import CROP_RATIO, CROP_SCALE, IMAGE_SIZE
from concertina.model.rng import SampleKeyedRNG


def resize_with_pad(images: Tensor, height: int, width: int, pad_value: float = 0.0) -> Tensor:
    """Aspect-preserving bilinear resize of `(N, C, H, W)` images followed by centered constant padding."""
    cur_height, cur_width = images.shape[-2:]
    if cur_height == height and cur_width == width:
        return images
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = F.interpolate(
        images.float(), size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    pad_height = height - resized_height
    pad_width = width - resized_width
    pad_top = pad_height // 2
    pad_left = pad_width // 2
    padded = F.pad(
        resized,
        (pad_left, pad_width - pad_left, pad_top, pad_height - pad_top),
        mode="constant",
        value=pad_value,
    )
    return padded.to(images.dtype)


def prepare_images(batch: dict, camera_keys: list[str]) -> Tensor:
    """Stack the configured cameras into `(B, C, H, W)` (one camera) or `(B, V, C, H, W)` float images."""
    images = []
    for key in camera_keys:
        if key not in batch:
            raise KeyError(f"Camera {key!r} is missing from the batch (available: {sorted(batch)})")
        image = batch[key]
        if image.ndim == 5:  # (B, T, C, H, W) with a single observation step
            image = image[:, -1]
        images.append(resize_with_pad(image, *IMAGE_SIZE))
    return torch.stack(images, dim=1) if len(images) > 1 else images[0]


def random_resized_crop(images: Tensor, rng: SampleKeyedRNG) -> Tensor:
    """Random resized crop of every camera image, returned at the input resolution.

    The crop of each image is drawn from the sample-keyed RNG; all crops are resampled by a single
    `grid_sample` call that reproduces `F.interpolate(crop, bilinear, align_corners=False)`.
    """
    leading_shape = images.shape[:-3]
    flat = images.reshape(-1, *images.shape[-3:])
    num_images, _, height, width = flat.shape
    views_per_sample = num_images // leading_shape[0]

    scale_uniform = rng.rand(
        (views_per_sample,), stream="image.random_resized_crop.scale", device="cpu"
    ).reshape(-1)
    ratio_uniform = rng.rand(
        (views_per_sample,), stream="image.random_resized_crop.ratio", device="cpu"
    ).reshape(-1)
    offset_uniform = rng.rand(
        (views_per_sample, 2), stream="image.random_resized_crop.offset", device="cpu"
    ).reshape(-1, 2)
    # Float64 box arithmetic matches Python-float crop sizes and offsets exactly.
    area = float(height * width)
    scale = CROP_SCALE[0] + (CROP_SCALE[1] - CROP_SCALE[0]) * scale_uniform.double()
    ratio = CROP_RATIO[0] + (CROP_RATIO[1] - CROP_RATIO[0]) * ratio_uniform.double()
    crop_heights = torch.sqrt(area * scale / ratio).round().clamp_(1, height).long()
    crop_widths = torch.sqrt(area * scale * ratio).round().clamp_(1, width).long()
    max_tops = height - crop_heights
    max_lefts = width - crop_widths
    tops = torch.minimum((offset_uniform[:, 0].double() * (max_tops + 1).double()).floor().long(), max_tops)
    lefts = torch.minimum(
        (offset_uniform[:, 1].double() * (max_lefts + 1).double()).floor().long(), max_lefts
    )

    device = flat.device
    tops, lefts = tops.to(device), lefts.to(device)
    crop_heights = crop_heights.to(device=device, dtype=torch.float32)
    crop_widths = crop_widths.to(device=device, dtype=torch.float32)
    output_y = torch.arange(height, device=device, dtype=torch.float32)
    output_x = torch.arange(width, device=device, dtype=torch.float32)
    # `align_corners=False` maps output pixel centers to `(out + 0.5) * crop / size - 0.5` in crop coordinates,
    # clamped to the crop.
    source_y = (output_y.unsqueeze(0) + 0.5) * crop_heights.unsqueeze(1) / height - 0.5
    source_x = (output_x.unsqueeze(0) + 0.5) * crop_widths.unsqueeze(1) / width - 0.5
    source_y = torch.minimum(
        source_y.clamp_min_(0.0), (crop_heights - 1.0).unsqueeze(1)
    ) + tops.float().unsqueeze(1)
    source_x = torch.minimum(
        source_x.clamp_min_(0.0), (crop_widths - 1.0).unsqueeze(1)
    ) + lefts.float().unsqueeze(1)
    grid_y = (2.0 * (source_y + 0.5) / height - 1.0).unsqueeze(2).expand(num_images, height, width)
    grid_x = (2.0 * (source_x + 0.5) / width - 1.0).unsqueeze(1).expand(num_images, height, width)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    cropped = F.grid_sample(flat.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return cropped.to(dtype=images.dtype).reshape(*leading_shape, *images.shape[-3:])


def mask_state_dims(
    state: Tensor,
    dims: list[int],
    probability: float,
    rng: SampleKeyedRNG,
    *,
    ratio_range: tuple[float, float] = (1.0, 1.0),
    fill_value: float = 0.0,
) -> Tensor:
    """With `probability` per sample, set a fraction drawn uniformly from `ratio_range` of `dims` (at least one,
    chosen at random) of the state to `fill_value`."""
    if probability <= 0.0 or not dims:
        return state
    batch_size = state.shape[0]
    apply = rng.rand((), stream="state_mask.apply", device=state.device) < float(probability)
    dims_tensor = torch.tensor(dims, device=state.device, dtype=torch.long)
    low, high = (float(value) for value in ratio_range)
    if low == 1.0 and high == 1.0:
        masked = state.clone()
        selected = masked[..., dims_tensor]
        masked[..., dims_tensor] = torch.where(
            apply.reshape((-1,) + (1,) * (selected.ndim - 1)), masked.new_tensor(float(fill_value)), selected
        )
        return masked
    if not bool(apply.any()):
        return state
    if low == high:
        ratios = torch.full((batch_size,), low, device=state.device, dtype=torch.float32)
    else:
        ratios = low + (high - low) * rng.rand((), stream="state_mask.ratio", device=state.device)
    num_dims = dims_tensor.numel()
    counts = torch.ceil(ratios * num_dims).to(torch.long)
    counts = torch.where(ratios > 0, torch.clamp(counts, min=1), counts)
    counts = torch.where(apply, torch.clamp(counts, min=0, max=num_dims), torch.zeros_like(counts))
    if not bool((counts > 0).any()):
        return state
    masked = state.clone()
    fill = masked.new_tensor(float(fill_value))
    for index in torch.nonzero(counts > 0, as_tuple=False).flatten().tolist():
        permutation = rng.randperm(index, num_dims, stream="state_mask.permutation", device=state.device)
        masked[index, ..., dims_tensor[permutation[: int(counts[index])]]] = fill
    return masked

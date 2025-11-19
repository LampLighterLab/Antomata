from collections.abc import Iterable
from typing import Callable

import torch
import torch.nn.functional as F


# 3x3 convolution kernel that counts the Moore neighborhood
_KERNEL = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32).view(
    1, 1, 3, 3
)


def grid_from_coords(
    coords: Iterable[tuple[int, int]],
    *,
    size: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a Life grid with 1.0 values at provided (row, column) coordinates."""
    grid = torch.zeros((size, size), dtype=dtype, device=device)
    for y, x in coords:
        grid[y, x] = 1.0
    return grid


def _to_float32(state: torch.Tensor) -> torch.Tensor:
    if state.dtype != torch.float32:
        return state.to(dtype=torch.float32)
    return state


def _reshape_to_bchw(state: torch.Tensor) -> torch.Tensor:
    """
    torch.conv2d requires an input of shape 4, with B, C, H, W.
    B (batch): how many separate grids/images you process in parallel.
    C (channels): per-cell feature planes. A standard Life board has one channel (alive/dead).
    H, W: height and width.
    """
    state_float = _to_float32(state)
    if state_float.dim() != 2:
        raise ValueError("State must have exactly 2 dimensions (H, W).")
    return state_float.unsqueeze(0).unsqueeze(0)


def _restore_shape(state_bchw: torch.Tensor) -> torch.Tensor:
    return state_bchw.squeeze(0).squeeze(0)


def convolve_neighbors(batched_state: torch.Tensor, kernel, wrap: bool) -> torch.Tensor:
    kernel = kernel.to(batched_state.device)
    if wrap:
        padded = F.pad(batched_state, (1, 1, 1, 1), mode="circular")
        neighbor_counts = F.conv2d(padded, kernel, padding = 0)
    else:
        neighbor_counts = F.conv2d(batched_state, kernel, padding=1)
        alive_mas = (batched_state > 0.5).to(neighbor_counts.dtype)
        neighbor_counts = neighbor_counts * alive_mas
        return neighbor_counts
    

def gaussian_neighbor_basis(neighbor_sums: torch.Tensor, sigma: float) -> torch.Tensor:
    centers = torch.arange(0, 9, device=neighbor_sums.device, dtype=neighbor_sums.dtype)
    centers = centers.view(1, 1, 1, 1, 9)
    return torch.exp(-0.5 * ((neighbor_sums.unsqueeze(-1) - centers) / sigma) ** 2)


def prepare_rule_weights(
    birth_weights: torch.Tensor | None,
    survival_weights: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if birth_weights is None:
        prepared_birth = torch.zeros(9, device=device, dtype=dtype)
        prepared_birth[3] = 1.0
    else:
        prepared_birth = birth_weights.to(device=device, dtype=dtype)

    if survival_weights is None:
        prepared_survival = torch.zeros(9, device=device, dtype=dtype)
        prepared_survival[2] = 1.0
        prepared_survival[3] = 1.0
    else:
        prepared_survival = survival_weights.to(device=device, dtype=dtype)

    return (
        prepared_birth.view(1, 1, 1, 1, 9),
        prepared_survival.view(1, 1, 1, 1, 9),
    )


def preactivation(
    x: torch.Tensor,
    birth_score: torch.Tensor,
    survival_score: torch.Tensor,
    *,
    bias: float = 0.5,
) -> torch.Tensor:
    return (1.0 - x) * birth_score + x * survival_score - bias


def build_activation(
    alpha: float, hard: bool
) -> Callable[[torch.Tensor], torch.Tensor]:
    if hard:

        def hard_activation(values: torch.Tensor) -> torch.Tensor:
            return (values > 0).to(values.dtype)

        return hard_activation

    def smooth_activation(values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(alpha * values)

    return smooth_activation


def life_step(state: torch.Tensor, wrap: bool = True) -> torch.Tensor:
    """
    Advance Conway's Game of Life by one tick using a 3x3 convolution kernel.

    Args:
        state: Tensor of shape (..., H, W) or (..., 1, H, W) with values in {0, 1}.
        wrap: When True use circular padding (toroidal surface). Otherwise zeros.

    Returns:
        Tensor with the same shape as the input containing the next state.
    """
    kernel = _KERNEL
    if state.dim() != 2:
        raise ValueError("State must have exactly 2 dimensions (H, W).")

    batch_state = _reshape_to_bchw(state)

    neighbor_counts = convolve_neighbors(batch_state, kernel, wrap)

    alive = batch_state > 0.5
    next_state = ((neighbor_counts == 3) | (alive & (neighbor_counts == 2))).to(
        batch_state.dtype
    )

    return _restore_shape(next_state)


def life_step_smooth(
    state: torch.Tensor,
    *,
    alpha: float = 8.0,
    sigma: float = 0.20,
    wrap: bool = True,
    hard: bool = False,
    birth_weights: torch.Tensor | None = None,
    survival_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Differentiable, single-threshold update that reduces to Birth:3/Survive:2-3 in the hard limit.

    A preactivation is formed from the neighbor sum s and current state x,
    then a single nonlinearity produces the next state. This avoids explicit
    boolean equality tests on s.

    preactivation = (1 - x) * birth_score(s) + x * survive_score(s) - 0.5
    next_state = sigmoid(alpha * z)  # or hard: y = (z > 0)

    Args:
        state: Tensor of shape (..., H, W) or (..., 1, H, W) with values in [0, 1].
        alpha: Steepness of the final activation; larger -> closer to hard rule.
        sigma: Width of the Gaussian basis centered at integer neighbor counts.
        wrap: Use circular padding (toroidal surface) if True, else zero padding.
        hard: If True, return a binary 0/1 next state using (z > 0).
        birth_weights: Optional length-9 weights for birth score over counts 0..8.
        survival_weights: Optional length-9 weights for survival score over counts 0..8.

    Returns:
        Tensor with same shape as the input; float in [0, 1] unless hard=True.
    """
    if state.dim() != 2:
        raise ValueError("State must have exactly 2 dimensions (H, W).")
    batched_state = _reshape_to_bchw(state)
    kernel = _KERNEL
    activation = build_activation(alpha, hard)
    rule_birth, rule_survival = prepare_rule_weights(
        birth_weights,
        survival_weights,
        device=batched_state.device,
        dtype=batched_state.dtype,
    )

    neighbor_sums = convolve_neighbors(batched_state, kernel, wrap)
    basis = gaussian_neighbor_basis(neighbor_sums, sigma)

    birth_score = (basis * rule_birth).sum(dim=-1)
    survival_score = (basis * rule_survival).sum(dim=-1)

    z = preactivation(batched_state, birth_score, survival_score, bias=0.5)

    next_state = activation(z)

    return _restore_shape(next_state)

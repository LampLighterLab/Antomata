import torch
import torch.nn.functional as F


# 3x3 convolution kernel that counts the Moore neighborhood
_KERNEL = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32).view(
    1, 1, 3, 3
)


def life_step(state: torch.Tensor, wrap: bool = True) -> torch.Tensor:
    """
    Advance Conway's Game of Life by one tick using a 3x3 convolution kernel.

    Args:
        state: Tensor of shape (..., H, W) or (..., 1, H, W) with values in {0, 1}.
        wrap: When True use circular padding (toroidal surface). Otherwise zeros.

    Returns:
        Tensor with the same shape as the input containing the next state.
    """
    if state.dtype != torch.float32:
        state = state.to(dtype=torch.float32)

    if state.dim() < 2:
        raise ValueError("State must have at least 2 dimensions (H, W).")

    # Ensure the tensor has shape (B, C, H, W) for conv2d.
    if state.dim() == 2:
        batch_state = state.unsqueeze(0).unsqueeze(0)
    elif state.dim() == 3:
        batch_state = state.unsqueeze(1)
    else:
        # Assume already (B, C, H, W)
        batch_state = state

    device_kernel = _KERNEL.to(batch_state.device)
    if wrap:
        padded = F.pad(batch_state, (1, 1, 1, 1), mode="circular")
        neighbor_counts = F.conv2d(padded, device_kernel, padding=0)
    else:
        neighbor_counts = F.conv2d(batch_state, device_kernel, padding=1)

    alive = batch_state > 0.5
    next_state = ((neighbor_counts == 3) | (alive & (neighbor_counts == 2))).to(
        batch_state.dtype
    )

    # Squeeze back to the original rank.
    if state.dim() == 2:
        return next_state.squeeze(0).squeeze(0)
    if state.dim() == 3:
        return next_state.squeeze(1)
    return next_state


def life_step_smooth(
    state: torch.Tensor,
    *,
    alpha: float = 8.0,
    sigma: float = 0.20,
    wrap: bool = True,
    hard: bool = False,
    w_birth: torch.Tensor | None = None,
    w_survive: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Differentiable, single-threshold update that reduces to B3/S23 in the hard limit.

    A preactivation z(s, x) is formed from the neighbor sum s and current state x,
    then a single nonlinearity produces the next state. This avoids explicit
    boolean equality tests on s.

    z = (1 - x) * birth_score(s) + x * survive_score(s) - 0.5
    y = sigmoid(alpha * z)  # or hard: y = (z > 0)

    Args:
        state: Tensor of shape (..., H, W) or (..., 1, H, W) with values in [0, 1].
        alpha: Steepness of the final activation; larger -> closer to hard rule.
        sigma: Width of the Gaussian basis centered at integer neighbor counts.
        wrap: Use circular padding (toroidal surface) if True, else zero padding.
        hard: If True, return a binary 0/1 next state using (z > 0).
        w_birth: Optional length-9 weights for birth score over counts 0..8.
        w_survive: Optional length-9 weights for survive score over counts 0..8.

    Returns:
        Tensor with same shape as input; float in [0,1] unless hard=True.
    """
    x = state.to(dtype=torch.float32)

    # Ensure BCHW shape for convolution
    if x.dim() == 2:
        x_bchw = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x_bchw = x.unsqueeze(1)
    else:
        x_bchw = x

    device_kernel = _KERNEL.to(x_bchw.device)
    if wrap:
        s = F.conv2d(
            F.pad(x_bchw, (1, 1, 1, 1), mode="circular"), device_kernel, padding=0
        )
    else:
        s = F.conv2d(x_bchw, device_kernel, padding=1)

    # Gaussian basis centered on integer counts 0..8
    centers = torch.arange(0, 9, device=s.device, dtype=s.dtype).view(1, 1, 1, 1, 9)
    phi = torch.exp(-0.5 * ((s.unsqueeze(-1) - centers) / sigma) ** 2)  # (..., 9)

    # Default to B3/S23 if weights not provided
    if w_birth is None:
        w_birth = torch.zeros(9, device=s.device, dtype=s.dtype)
        w_birth[3] = 1.0
    if w_survive is None:
        w_survive = torch.zeros(9, device=s.device, dtype=s.dtype)
        w_survive[2] = 1.0
        w_survive[3] = 1.0

    w_birth = w_birth.view(1, 1, 1, 1, 9)
    w_survive = w_survive.view(1, 1, 1, 1, 9)
    birth_score = (phi * w_birth).sum(dim=-1)  # (..., H, W)
    survive_score = (phi * w_survive).sum(dim=-1)  # (..., H, W)

    z = (1.0 - x_bchw) * birth_score + x_bchw * survive_score
    z = z - 0.5

    y = torch.sigmoid(alpha * z) if not hard else (z > 0).to(z.dtype)

    # Return to original rank
    if state.dim() == 2:
        return y.squeeze(0).squeeze(0)
    if state.dim() == 3:
        return y.squeeze(1)
    return y

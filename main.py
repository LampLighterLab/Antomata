from time import sleep

import torch

from config import get_active_config
from life import life_step, life_step_smooth


def render_binary(state: torch.Tensor) -> str:
    """Render a 0/1 tensor grid to ASCII blocks."""
    s = (state > 0.5).to(dtype=torch.int32)
    rows = ["".join("██" if cell else "  " for cell in row) for row in s]
    return "\n".join(rows)


def render_soft(state: torch.Tensor) -> str:
    """Render a [0,1] tensor grid with coarse grayscale blocks."""
    # Map intensity to characters: light -> dark
    shades = "  ░▒▓█"
    s = state.clamp(0, 1).cpu().numpy()
    rows = []
    for row in s:
        chars = []
        for val in row:
            idx = int(round(float(val) * (len(shades) - 1)))
            ch = shades[idx]
            chars.append(ch + ch)  # double-width cell
        rows.append("".join(chars))
    return "\n".join(rows)


def main():
    cfg = get_active_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = int(cfg.get("size", 20))
    steps = int(cfg.get("steps", 10))
    wrap = bool(cfg.get("wrap", True))
    mode = cfg.get("mode", "classic")

    use_smooth = mode == "smooth"
    smooth_cfg = cfg.get("smooth") if isinstance(cfg.get("smooth"), dict) else {}
    hard = bool(smooth_cfg.get("hard", False))
    alpha = float(smooth_cfg.get("alpha", 8.0))
    sigma = float(smooth_cfg.get("sigma", 0.20))

    grid = torch.zeros((size, size), dtype=torch.float32, device=device)

    # Simple glider pattern.
    glider = [(1, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    for y, x in glider:
        grid[y, x] = 1.0

    state = grid
    display_soft = use_smooth and not hard
    for tick in range(steps + 1):
        print(f"Step {tick}")
        if display_soft:
            print(render_soft(state))
        else:
            print(render_binary(state))
        print()
        sleep(0.05)
        if tick < steps:
            if use_smooth:
                state = life_step_smooth(
                    state,
                    alpha=alpha,
                    sigma=sigma,
                    wrap=wrap,
                    hard=hard,
                )
            else:
                state = life_step(state, wrap=wrap)


if __name__ == "__main__":
    main()

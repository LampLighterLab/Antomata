
import torch

from life import grid_from_coords, life_step, life_step_smooth


def test_smooth_mode_parameter_showcase():
    """Document the effect of configurable knobs in the smooth Life variant."""
    # A cross pattern exercises births, survival, and death transitions.
    seed = grid_from_coords(
        [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)],
        size=6,
    )

    # Hard mode with large alpha collapses to the discrete Conway rule.
    classic = life_step(seed, wrap=True)
    smooth_hard = life_step_smooth(seed, wrap=True, hard=True, alpha=12.0)
    assert torch.equal(classic, smooth_hard)

    # Softer activations keep fractional states that are useful for experimentation.
    smooth_soft = life_step_smooth(seed, wrap=True, hard=False, alpha=1.5, sigma=0.6)
    assert torch.any((smooth_soft > 0.0) & (smooth_soft < 1.0))

    # Cranking alpha back up (while staying in soft mode) produces near-binary outputs.
    smooth_sharp = life_step_smooth(seed, wrap=True, hard=False, alpha=30.0, sigma=0.2)
    assert torch.all(
        (smooth_sharp < 1e-3) | (smooth_sharp > 1.0 - 1e-3)
    ), "Large alpha should mimic the hard threshold."

    # Custom rule weights let us bias the automaton. Here every cell dies because
    # we zero the survival weights—a dramatic change compared to the default rule.
    extinction = life_step_smooth(
        seed,
        wrap=True,
        hard=True,
        alpha=12.0,
        survival_weights=torch.zeros(9, dtype=torch.float32),
    )
    # Survival weights at 0 force every previously alive cell to die, even if
    # births may still occur elsewhere.
    assert torch.all(extinction[seed > 0.5] == 0)
    assert not torch.equal(classic, extinction)

    # The wrap flag controls whether we treat the grid as a torus (neighbors wrap).
    # A horizontal line on the top edge spawns a new cell across the bottom when wrap=True.
    top_edge_line = grid_from_coords([(0, 1), (0, 2), (0, 3)], size=5)
    next_wrap = life_step(top_edge_line, wrap=True)
    next_no_wrap = life_step(top_edge_line, wrap=False)
    assert next_wrap[4, 2].item() == 1.0
    assert next_no_wrap[4, 2].item() == 0.0


def test_beehive_is_stable():
    """Beehive still life should remain unchanged between generations."""
    beehive = grid_from_coords(
        [(1, 2), (1, 3), (2, 1), (2, 4), (3, 2), (3, 3)],
        size=6,
    )
    assert torch.equal(beehive, life_step(beehive, wrap=False))


def test_blinker_oscillates_with_period_two():
    """The blinker should swap between vertical and horizontal every tick."""
    horizontal = grid_from_coords(
        [(2, 1), (2, 2), (2, 3)],
        size=5,
    )
    vertical = grid_from_coords(
        [(1, 2), (2, 2), (3, 2)],
        size=5,
    )

    next_state = life_step(horizontal, wrap=False)
    assert torch.equal(next_state, vertical)
    assert torch.equal(life_step(next_state, wrap=False), horizontal)


def test_glider_advances_diagonally():
    """After four steps the glider should shift one cell down and to the right."""
    glider_start = grid_from_coords(
        [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        size=6,
    )
    expected = grid_from_coords(
        [(1, 2), (2, 3), (3, 1), (3, 2), (3, 3)],
        size=6,
    )

    state = glider_start.clone()
    for _ in range(4):
        state = life_step(state, wrap=False)

    assert torch.equal(state, expected)

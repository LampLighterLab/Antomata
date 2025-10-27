# Ant-omata

Cellular Automata can be viewed as convolutions, and this repo aims to use this model to learn simple local interaction rules that can reliably converge to specific patterns, namely of ant nests. We use Conway's game of life as a starting point, creating a smooth, differentiable rule that can be optimized over.

### Getting Started
[uv](https://docs.astral.sh/uv/getting-started/installation/) is recommended.
- Create a virtual environment and sync deps from `pyproject.toml`/`uv.lock`:
  - `uv venv` (creates `.venv`)
  - `source .venv/bin/activate` (Unix/macOS) or `.venv\Scripts\activate` (Windows)
  - `uv sync`

Alternatively, you can run ad‑hoc without activating the venv:
- `uv run python main.py`
- `uv run pytest tests`

#### Running the Demo
- Default demo uses a glider pattern and the active profile from `config.py`.
- Run:
  - `uv run python main.py`

#### Configuration
- Profiles live in `config.py`. Set `ACTIVE_PROFILE` to switch:
  - `classic_demo`: binary B3/S23
  - `smooth_demo`: differentiable update

### Testing
- Run the suite:
  - `uv run pytest tests`
- What’s covered:
  - Parameter showcase for the smooth update (alpha/sigma/hard/weights/wrap)
  - Still life (beehive) stability
  - Oscillator (blinker) period‑2 behavior
  - Glider diagonal movement over 4 ticks

### Performance
- Uses GPU if `torch.cuda.is_available()` is true; otherwise runs on CPU.


### Project Layout
- `life.py` — core logic: neighbor convolution, classic/smooth steps, `grid_from_coords`
- `main.py` — CLI demo and ASCII rendering
- `config.py` — profiles and runtime parameters
- `tests/` — pytest suite for parameter demos and regressions

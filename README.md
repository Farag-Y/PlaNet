PlaNet
======

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

> **2026 Update:** This is an updated, working fork of the original [Kaixhin/PlaNet](https://github.com/Kaixhin/PlaNet) repository. The original codebase relied on gym 0.21 and Python tooling from 2021 that can no longer be installed with modern package managers. This fork fixes compatibility with Python 3.9, gym 0.23, NumPy 1.x, and PyTorch 2.x, and migrates the project setup to [uv](https://docs.astral.sh/uv/).

PlaNet: A Deep Planning Network for Reinforcement Learning [[1]](#references). Supports symbolic/visual observation spaces and some Gym environments (including classic control/non-MuJoCo environments — DeepMind Control Suite/MuJoCo are optional). Hyperparameters are tuned for DeepMind Control Suite and would need adjustment for other domains.

Run with `python main.py`. For best performance with DeepMind Control Suite, set `MUJOCO_GL=egl` (see [dm_control rendering docs](https://github.com/deepmind/dm_control#rendering)).

Results and pretrained models can be found in the [releases](https://github.com/Farag-Y/PlaNet/releases).


Installation
------------

**Requirements:** Python 3.9+, [uv](https://docs.astral.sh/uv/)

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo and set up the environment in one step:

```bash
git clone https://github.com/Farag-Y/PlaNet
cd PlaNet
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, creates a `.venv` automatically, and installs all pinned dependencies — no manual venv creation needed.

Activate the environment and run:

```bash
source .venv/bin/activate
python main.py
```

Or run directly through uv without activating:

```bash
uv run python main.py
```

### Optional: DeepMind Control Suite

Install [dm_control](https://github.com/deepmind/dm_control) separately if you want to use Control Suite environments:

```bash
uv pip install dm_control
```


Dependencies
------------

| Package | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | |
| PyTorch | 2.x | MPS (Apple Silicon) supported |
| torchvision | 0.23+ | |
| gym | 0.23.1 | Pinned — newer versions break the step API |
| numpy | < 2.0 | Pinned for gym 0.23 compatibility |
| opencv-python | latest | |
| plotly | latest | |
| pygame | latest | Required by gym classic control renderers |
| tqdm | latest | |

> **Note on gym versions:** gym ≥ 0.26 changed the `step()` return from a 4-tuple to a 5-tuple and renamed several environments. This codebase uses the 0.23.x API. `Pendulum-v0` has been updated to `Pendulum-v1` (renamed in gym 0.22).


Links
-----

- [Introducing PlaNet: A Deep Planning Network for Reinforcement Learning](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html)
- [google-research/planet](https://github.com/google-research/planet)
- [Original Kaixhin/PlaNet](https://github.com/Kaixhin/PlaNet)


Acknowledgements
----------------

- [@Kaixhin](https://github.com/Kaixhin) for the original PyTorch implementation
- [@danijar](https://github.com/danijar) for [google-research/planet](https://github.com/google-research/planet) and [help reproducing results](https://github.com/google-research/planet/issues/28)
- [@sg2](https://github.com/sg2) for [running experiments](https://github.com/Kaixhin/PlaNet/issues/9)
- [@JanS97](https://github.com/JanS97) for [fixing the latent overshooting objective](https://github.com/Kaixhin/PlaNet/pull/30)


References
----------

[1] [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)

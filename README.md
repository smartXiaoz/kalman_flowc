# FlowCritic

## Paper

**FlowCritic: Bridging Value Estimation with Flow Matching in Reinforcement Learning** introduces a generative value-estimation paradigm for reinforcement learning. Instead of regressing a single scalar value, FlowCritic learns a velocity field that transports a tractable prior distribution to a bootstrapped return distribution. ODE integration can then generate value samples conditioned on a state.

The generated samples provide distributional information during policy optimization. FlowCritic uses their coefficient of variation as a measure of estimation noise, giving more influence to reliable samples and reducing the effect of uncertain samples. Truncated sampling and velocity-field clipping are included to improve the robustness of value learning.

The paper studies the method on continuous-control simulation tasks and validates a trained policy on a quadrupedal robot. This repository contains the training implementation and a small run-analysis utility.

## Repository

```text
src/flow_ppo_single2.py        FlowCritic training entry point
src/flow_matching.py           Flow-matching value model and sampling code
config/config_*.json           Environment configuration files
scripts/score_plot_protocol.py Run-analysis utility
environment.yml                Reproducible conda environment
```

Isaac Gym, IsaacGymEnvs, their simulator binaries, and their task assets are external dependencies and are not included in this repository.

## Installation

The repository includes `environment.yml`, so a new user does not need an existing environment named `RLS`.

### 1. Create the conda environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate flowcritic
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The environment file installs the Python, PyTorch, CUDA runtime, and Python packages used by the release. A compatible NVIDIA driver must already be installed on the host.

### 2. Install Isaac Gym Preview 4

Isaac Gym Preview 4 contains NVIDIA binaries and is not distributed through PyPI or this repository. Download it from the [NVIDIA Isaac Gym page](https://developer.nvidia.com/isaac-gym), accept NVIDIA's license, and extract it locally. Then install its Python package into the new environment:

```bash
pip install -e /path/to/isaacgym/python
```

Here `/path/to/isaacgym` is the extracted directory containing `python/setup.py`.

### 3. Install IsaacGymEnvs

Install the task package from NVIDIA's repository:

```bash
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
cd IsaacGymEnvs
git checkout aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3
pip install -e .
cd ..
```

The pinned revision is the IsaacGymEnvs 1.5.1 source used with this release.

### 4. Verify the installation

Isaac Gym must be imported before PyTorch:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python - <<'PY'
import isaacgym
import isaacgymenvs
import torch

print("Isaac Gym and IsaacGymEnvs imported successfully")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

Before running FlowCritic, also run an Isaac Gym example from the extracted package as recommended by NVIDIA. If the example cannot start, resolve the simulator, driver, or Vulkan installation first.

## Run Training

Choose an environment configuration from `config/` and run the training entry point from the repository root:

```bash
conda activate flowcritic
python -u src/flow_ppo_single2.py \
  --config-path config/config_Ant.json
```

Use a separate copy of the configuration when running independent repetitions. The entry point accepts a GPU selector and optional paths for the run name and training logs; these options are useful when launching several jobs through an external scheduler.

The configuration files are ordinary JSON and are loaded without modification by the training script. Isaac Gym task names must match the environment names supported by the installed IsaacGymEnvs package.

## Inspect a Run

The analysis utility reads a completed round directory and writes Markdown, CSV, and JSON summaries:

```bash
python scripts/score_plot_protocol.py <round-directory> \
  --output-dir <output-directory>
```

A round directory contains a manifest and one run directory per job. Each run directory should keep the resolved configuration and the logs produced by the training entry point. The utility is read-only with respect to the training source code.

## Citation

- [IEEE Computer Society Digital Library](https://doi.ieeecomputersociety.org/10.1109/TPAMI.2026.3728663)
- [DOI resolver](https://doi.org/10.1109/TPAMI.2026.3728663)

```bibtex
@ARTICLE{11672875,
  author={Zhong, Shan and Ding, Shutong and Diao, He and Wang, Xiangyu and Teh, Kah Chan and Peng, Bei},
  journal={IEEE Transactions on Pattern Analysis & Machine Intelligence},
  title={FlowCritic: Bridging Value Estimation with Flow Matching in Reinforcement Learning},
  year={2026},
  volume={},
  number={01},
  pages={1-17},
  doi={10.1109/TPAMI.2026.3728663},
  
}
```

## Assets

The root-level MuJoCo XML files were not referenced by the release source or configuration and are not part of this repository. The included training path uses Isaac Gym task assets supplied by the external simulator package.

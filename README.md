# SO-101 Reach

**Goal**: Tune the training environment to make the reach policy work!

## Setup

Make accounts (or use existing accounts):

- [Tailscale](https://tailscale.com/) (use a non-utexas.edu email)
    - Follow the instructions on account creation to install Tailscale on your local machine
    - When prompted to add a second device, skip the tutorial
- [Github](https://github.com/signup)
- [wandb](https://wandb.ai/signup) (select Models under "What do you want to try first?", if prompted)

Install:

- [Visual Studio Code](https://code.visualstudio.com/download) or an IDE of your choosing
- [git](https://git-scm.com/install/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Add the server machine to your Tailnet (get link from class recording, 11/13).

## Installing dependencies

Fork this repository, then clone your forked repository. Also clone the [robot_rl feature/barebones branch](https://github.com/KyleM73/robot_rl/tree/feature/barebones).

```bash
git clone https://github.com/<username>/so101_isaac.git
git clone https://github.com/KyleM73/robot_rl.git
```


Fill out `scripts/local_ray/.env.ray` and `scripts/local_ray/job_config.yaml`. Use absolute paths (should start with `/home/<user>/...`)

> A couple of notes:
> - All commands below are run from the `so101/scripts` directory.
> - If you are in a conda environment, deactivate it first with `conda deactivate`.
> - Some of the commands below may be different on Windows. See [below](#additional-referencesdocumentation) for command equivalents.

Install Python dependencies:

```bash
uv sync
```

Activate the uv environment. You will need to do this each time you open a new shell, before running any of the scripts.

```bash
source .venv/bin/activate
```

Ensure connectivity to the server with

```bash
./ray.sh list
```

This should display a blank table.

## Using the ray.sh interface

### `./ray.sh job`

- Sends a job to the server.
    - You can modify the script that runs (e.g. between `train.py` and `play.py`) in the `python_script` field of `job_config.yaml`
- Can be followed by any arguments you'd like to pass to the script (e.g. `--task reach-v0`).

### `./ray.sh stop <job_id>`

- Stop a running job
- Can provide additional arguments (see [`ray job stop` docs](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/cli.html#ray-job-stop))

### `./ray.sh list`

- List your currently running jobs, ascending by start time
- View all users' runs with `--all_users`
- View the status of all runs with `--all_statuses`

### `./ray.sh logs <job_id> <out_file>`

- Download logs for a job and write them to a specified file
- Generally, you can use W&B to view your run logs and metrics. This function is mainly for when your job fails before it can deploy, or if you aren't using W&B

## Relevant files

- `tasks/reach_env_cfg.py`: the main environment config
- `agents/rsl_rl_ppo_cfg.py`: config for the PPO runner
- `mdp/rewards.py`: definitions for reward functions
- `scripts/train.py` and `scripts/play.py`: train and play scripts

## Other things

When you commit changes to your branch, DON'T push the changes in .env.ray (never push API keys to Github).

## Additional references/documentation:

- [git docs](https://git-scm.com/docs)
- [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/index.html)
- [PowerShell equivalents for common Linux/bash commands](https://mathieubuisson.github.io/powershell-linux-bash/)
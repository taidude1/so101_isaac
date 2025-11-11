import os

TASK_DIR = os.path.dirname(__file__)

import gymnasium as gym

from . import agents
from . import tasks

##
# Register Gym environments.
##


gym.register(
    id="reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{tasks.__name__}.reach_env_cfg:ReachTaskCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ReachPPORunnerCfg",
    },
)
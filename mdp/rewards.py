from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def pose_time_based(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float = 1.0,
    duration: float = 1.0,
    command_name: str = "ee_pose",
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    pos_error = torch.norm(curr_pos_w - des_pos_w, dim=1)
    pos_reward = torch.exp(-pos_error / std)
    # des_quat_b = command[:, 3:7]
    # des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    # curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    # ori_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    # ori_reward = torch.exp(-ori_error / std)
    # time_elapsed = env.episode_length_buf * env.step_dt
    return pos_reward
    # return torch.where(
    #     env.max_episode_length_s - time_elapsed < duration,
    #     pos_reward * (1 + ori_reward) / 2,
    #     pos_reward,
    #     # 0.0 
    # )


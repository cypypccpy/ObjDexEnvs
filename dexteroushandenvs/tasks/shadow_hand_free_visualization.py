# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from matplotlib.pyplot import axis
# import high_level_planner.pybullet_ik_solver
import numpy as np
import os
import random
import torch
import pickle

from utils.torch_jit_utils import *

# from isaacgym.torch_utils import *

from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

import matplotlib.pyplot as plt
from PIL import Image as Im
from utils import o3dviewer
import cv2
from torch import nn
import torch.nn.functional as F
import math
from scipy.interpolate import interp1d
from collections import deque

class ShadowHandFreeVisualization(BaseTask):
    def __init__(
        self,
        cfg,
        sim_params,
        physics_engine,
        device_type,
        device_id,
        headless,
        agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]],
        is_multi_agent=False,
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.allegro_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time / (control_freq_inv * self.sim_params.dt))
            )
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in [
            "block",
            "egg",
            "pen",
            "ycb/banana",
            "ycb/can",
            "ycb/mug",
            "ycb/brick",
        ]

        self.ignore_z = self.object_type == "pen"

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "ball": "urdf/objects/ball.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "ycb/banana": "urdf/ycb/011_banana/011_banana.urdf",
            "ycb/can": "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
            "ycb/mug": "urdf/ycb/025_mug/025_mug.urdf",
            "ycb/brick": "urdf/ycb/061_foam_brick/061_foam_brick.urdf",
        }

        self.asset_files_dict = {
            "box": "arctic_assets/object_urdf/box.urdf",
            "scissors": "arctic_assets/object_urdf/scissors.urdf",
            "microwave": "arctic_assets/object_urdf/microwave.urdf",
            "laptop": "arctic_assets/object_urdf/laptop.urdf",
            "capsulemachine": "arctic_assets/object_urdf/capsulemachine.urdf",
            "ketchup": "arctic_assets/object_urdf/ketchup.urdf",
            "mixer": "arctic_assets/object_urdf/mixer.urdf",
            "notebook": "arctic_assets/object_urdf/notebook.urdf",
            "phone": "arctic_assets/object_urdf/phone.urdf",
            "waffleiron": "arctic_assets/object_urdf/waffleiron.urdf",
            "espressomachine": "arctic_assets/object_urdf/espressomachine.urdf",
        }

        self.used_training_objects = [
            "box",
            "capsulemachine",
            "espressomachine",
            "ketchup",
            "laptop",
            "microwave",
            "mixer",
            "notebook",
            "phone",
            "scissors",
            "waffleiron",
        ]
        # self.used_training_objects = ['ball', "block", "pen", "obj0", "obj1", "obj2", "obj4", "obj6", "obj7", "obj9", "obj10"]
        self.used_training_objects = self.cfg["env"]["used_training_objects"]
        self.used_hand_type = self.cfg["env"]["used_hand_type"]
        self.traj_index = self.cfg["env"]["traj_index"]
        
        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        # if not (self.obs_type in ["point_cloud", "full_state"]):
        #     raise Exception(
        #         "Unknown type of observations!\nobservationType should be one of: [point_cloud, full_state]")

        print("Obs type:", self.obs_type)

        self.num_point_cloud_feature_dim = 384
        self.one_frame_num_obs = 178
        self.num_obs_dict = {
            "full_state": 364,
        }
        # self.num_obs_dict = {
        #     "point_cloud": 111 + self.num_point_cloud_feature_dim * 3,
        #     "point_cloud_for_distill": 111 + self.num_point_cloud_feature_dim * 3,
        #     "full_state": 111
        # }
        self.contact_sensor_names = ["ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle", "thdistal"]

        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            # num_states = 215 + 384 * 3
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 24

        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 48

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.enable_camera_sensors = self.cfg["env"]["enableCameraSensors"]
        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)
        self.num_envs = cfg["env"]["numEnvs"]

        if self.point_cloud_debug:
            import open3d as o3d
            from utils.o3dviewer import PointcloudVisualizer

            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()
        else:
            self.pointCloudVisualizer = None

        super().__init__(cfg=self.cfg, enable_camera_sensors=self.enable_camera_sensors)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.5, -0.0, 1.2)
            cam_target = gymapi.Vec3(-0.5, -0.0, 0.2)

            # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # cam_pos = gymapi.Vec3(self.obj_params[1, 4]+ 0.5, self.obj_params[1, 5] + 0.5, self.obj_params[1, 6] + 0.5 + 0.5)
            # cam_target = gymapi.Vec3(self.obj_params[1, 4], self.obj_params[1, 5], self.obj_params[1, 6])

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.jacobian_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_jacobian_tensor(self.sim, "hand")
        )
        self.another_jacobian_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_jacobian_tensor(self.sim, "another_hand")
        )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # print(self.trans_l[0])
        # print(self.trans_r[0])
        # print(self.obj_params[0, 4:7])
        # exit()

        # create some wrapper tensors for different slices
        self.another_allegro_hand_default_dof_pos = torch.zeros(
            self.num_allegro_hand_dofs, dtype=torch.float, device=self.device
        )
        # self.another_allegro_hand_default_dof_pos[:6] = to_torch([self.trans_l[1, 0], self.trans_l[1, 1], self.trans_l[1, 2],
        #                                                       self.rot_l[1, 0], self.rot_l[1, 1], self.rot_l[1, 2]], dtype=torch.float, device=self.device)
        # self.another_allegro_hand_default_dof_pos = to_torch([0.0, -0.49826458111314524, -0.01990020486871322, -2.4732269941140346, -0.01307073642274261, 2.00396583422025, 1.5480939705504309,
        #                                 0.0, -0.174, 0.785, 0.785,
        #                             0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)

        # self.allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        # self.allegro_hand_default_dof_pos[:6] = to_torch([self.trans_r[1, 0], self.trans_r[1, 1], self.trans_r[1, 2],
        #                                             self.rot_r[1, 0], self.rot_r[1, 1], self.rot_r[1, 2]], dtype=torch.float, device=self.device)
        self.allegro_hand_default_dof_pos = torch.zeros(
            self.num_allegro_hand_dofs, dtype=torch.float, device=self.device
        )
        # self.allegro_hand_default_dof_pos = to_torch([0.0, -0.49826458111314524, -0.01990020486871322, -2.4732269941140346, -0.01307073642274261, 2.00396583422025, 1.5480939705504309,
        #                                 0.0, -0.174, 0.785, 0.785,
        #                             0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)

        # print(self.obj_params[0, 6] / 1000)
        # print(self.trans_l[0, 2])
        # exit()

        self.object_default_dof_pos = to_torch(
            [self.obj_params[0, 0, 0]], dtype=torch.float, device=self.device
        )
        # self.object_default_dof_pos = torch.zeros(1, dtype=torch.float, device=self.device)
        # self.object_default_dof_pos = torch.zeros(7, dtype=torch.float, device=self.device)

        ## hand put
        # self.allegro_hand_default_dof_pos[6:] = to_torch([0,0,0.7,1.2,0,0.7,0.3,1.2, 0,0,0.7,1.2,0,0,0.7,1.2,], dtype=torch.float, device=self.device)

        ## hand grip
        # self.allegro_hand_default_dof_pos[6:] = to_torch([0,0.5,0.7,1.2,1.57,0.3,1.2,0.7,0,0.3,0.7,1.2,0,0.5,0.7,1.2,], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_allegro_hand_dofs
        ]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.allegro_hand_another_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, self.num_allegro_hand_dofs : self.num_allegro_hand_dofs * 2
        ]
        self.allegro_hand_another_dof_pos = self.allegro_hand_another_dof_state[..., 0]
        self.allegro_hand_another_dof_vel = self.allegro_hand_another_dof_state[..., 1]

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, self.num_allegro_hand_dofs * 2 : self.num_allegro_hand_dofs * 2 + 1
        ]
        self.object_dof_pos = self.object_dof_state[..., 0]
        self.object_dof_vel = self.object_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()

        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.object_init_quat = torch.zeros(
            (self.num_envs, 4), dtype=torch.float, device=self.device
        )

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.object_pose_for_open_loop = torch.zeros_like(
            self.root_state_tensor[self.object_indices, 0:7]
        )

        self.total_successes = 0
        self.total_resets = 0

        self.state_buf_stack_frames = []
        self.obs_buf_stack_frames = []

        for i in range(3):
            self.obs_buf_stack_frames.append(
                torch.zeros_like(self.obs_buf[:, 0 : self.one_frame_num_obs])
            )
            self.state_buf_stack_frames.append(torch.zeros_like(self.states_buf[:, 0:215]))

        self.object_seq_len = 20
        self.object_state_stack_frames = torch.zeros(
            (self.num_envs, self.object_seq_len * 3), dtype=torch.float, device=self.device
        )

        self.proprioception_close_loop = torch.zeros_like(self.allegro_hand_dof_pos[:, 0:22])
        self.another_hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.another_hand_indices[0], "panda_link7", gymapi.DOMAIN_ENV
        )
        print("another_hand_base_rigid_body_index: ", self.another_hand_base_rigid_body_index)
        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.hand_indices[0], "panda_link7", gymapi.DOMAIN_ENV
        )
        print("hand_base_rigid_body_index: ", self.hand_base_rigid_body_index)
        # with open("./demo_throw.pkl", "rb") as f:
        #     self.demo_throw = pickle.load(f)

        # print(self.demo_throw)
        # # self.demo_throw = to_torch(self.demo_throw['qpos'], dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        # self.demo_throw = to_torch(self.demo_throw['qpos'], dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )
        object_rb_count = self.gym.get_asset_rigid_body_count(self.object_asset)
        self.object_rb_handles = 94
        self.perturb_direction = torch_rand_float(
            -1, 1, (self.num_envs, 6), device=self.device
        ).squeeze(-1)

        self.predict_pose = self.goal_init_state[:, 0:3].clone()

        self.apply_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float
        )

        self.r_pos_global_init = self.trans_r[:, 0].clone()
        self.r_rot_global_init = self.rot_r_quat[:, 0].clone()
        self.l_pos_global_init = self.trans_l[:, 0].clone()
        self.l_rot_global_init = self.rot_l_quat[:, 0].clone()
        self.obj_pos_global_init = self.obj_params[:, 0, 4:7]
        self.obj_rot_global_init = self.obj_rot_quat[:, 0, 0:4]

        self.max_episode_length = self.trans_r.shape[0] - 10
        self.init_step_buf = torch.zeros_like(self.progress_buf)
        self.end_step_buf = torch.zeros_like(self.progress_buf)

        self.last_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
        )

        self.one_step_obs = torch.zeros(
            (self.num_envs, 80), dtype=torch.float32, device=self.device
        )
        self.obs_buffer = torch.zeros(
            (self.num_envs, 3, 80), dtype=torch.float32, device=self.device
        )

        self.ran = torch.zeros((self.num_envs, 14), dtype=torch.float32, device=self.device)

        self.test_high_level_planner = False
        if self.test_high_level_planner:
            import robomimic.utils.file_utils as FileUtils
            
            self.high_level_planner = "bc_trans"
            self.is_multi_object = True
            if self.is_multi_object:
                if self.high_level_planner == "bc_trans":
                    self.policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path="/home/user/robomimic/robomimic/../bc_transformer_trained_models/test/20240528172618/models/model_epoch_20000.pth", device=self.device, verbose=True)
                    
                    self.traj_len = 10
                    
                    obs = {"obj_joint": self.obj_params[0, 0:self.traj_len, 0:1].reshape(-1), "obj_trans": self.obj_params[0, 0:self.traj_len, 4:7].reshape(-1), "obj_quat": self.obj_rot_quat[0, 0:self.traj_len, 0:4].reshape(-1), "obj_bps": torch.tensor([0], dtype=torch.float, device=self.device).repeat((self.traj_len, 1)).reshape(-1)}
                    
                    self.timestep = 0  # always zero regardless of timestep type
                    self.update_obs(obs, reset=True)
                    self.obs_history = self._get_initial_obs_history(init_obs=obs)
                    self.obs = self._get_stacked_obs_from_history()

                if self.high_level_planner == "bc":
                    self.policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path="/home/user/DexterousHandEnvs/dexteroushandenvs/high_level_planner/results/bc_mo_model_epoch_5000.pth", device=self.device, verbose=True)
                    self.traj_len = 10
                    
            else:
                if self.high_level_planner == "bc_trans":
                    self.policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path="/home/user/DexterousHandEnvs/dexteroushandenvs/high_level_planner/results/bctrans_box_model_epoch_10000.pth", device=self.device, verbose=True)
                    
                    self.traj_len = 3
                    
                    obs = {"obj_joint": self.obj_params[0, 0:self.traj_len, 0:1].reshape(-1), "obj_trans": self.obj_params[0, 0:self.traj_len, 4:7].reshape(-1), "obj_quat": self.obj_rot_quat[0, 0:self.traj_len, 0:4].reshape(-1)}
                    
                    self.timestep = 0  # always zero regardless of timestep type
                    self.update_obs(obs, reset=True)
                    self.obs_history = self._get_initial_obs_history(init_obs=obs)
                    self.obs = self._get_stacked_obs_from_history()
                
                    # act = self.policy(ob=self.obs)
                    # print(act.shape)

                    # self.update_obs(obs, action=act, reset=False)
                    # for k in obs:
                    #     self.obs_history[k].append(obs[k][None])
                    # self.obs = self._get_stacked_obs_from_history()

                    # rot_r_quat = act[0:self.traj_len*4]
                    # trans_r = act[self.traj_len*4:self.traj_len*(4+3)]
                    # rot_l_quat = act[self.traj_len*(4+3):self.traj_len*(4+3+4)]
                    # trans_l = act[self.traj_len*(4+3+4):self.traj_len*(4+3+4+3)]

                    # print(trans_r)
                    # exit()
                if self.high_level_planner == "bc":
                    self.traj_len = 3
                    self.policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path="/home/user/DexterousHandEnvs/dexteroushandenvs/high_level_planner/results/bc_box_model_epoch_10000.pth", device=self.device, verbose=True)
                    # obs = {"obj_joint": self.obj_params[0, 0:self.traj_len, 0:1].reshape(-1), "obj_trans": self.obj_params[0, 0:self.traj_len, 4:7].reshape(-1), "obj_quat": self.obj_rot_quat[0, 0:self.traj_len, 0:4].reshape(-1)}

            
        self.if_calc_evaluation_metric = True
        self.position_error_tensor = torch.zeros(
            (self.num_envs, 489), dtype=torch.float32, device=self.device
        )
        self.rotation_error_tensor = torch.zeros(
            (self.num_envs, 489), dtype=torch.float32, device=self.device
        )
        
        if "mano" in self.used_hand_type:
            if self.used_hand_type == "mano_shadow":
                if self.use_joint_space_retargeting:
                    from high_level_planner.anyteleop_solver import TestOptimizer
                    self.ik_solver = TestOptimizer(hand_type="shadow")
                else:
                    from high_level_planner.pybullet_ik_solver import PybulletIKSolver
                    self.ik_solver = PybulletIKSolver("/home/user/DexterousHandEnvs/assets/urdf/dex-urdf/robots/hands/shadow_hand/shadow_hand_left_for_ik.urdf", "/home/user/DexterousHandEnvs/assets/urdf/dex-urdf/robots/hands/shadow_hand/shadow_hand_right_for_ik.urdf", hand_type="shadow")
                    
                    
            else:
                from high_level_planner.pybullet_ik_solver import PybulletIKSolver
                self.ik_solver = PybulletIKSolver("/home/user/DexterousHandEnvs/assets/urdf/mano-urdf/urdf/mano_left_fixed.urdf", "/home/user/DexterousHandEnvs/assets/urdf/mano-urdf/urdf/mano_right_fixed.urdf", hand_type="mano")

        
    def _get_initial_obs_history(self, init_obs):
        """
        Helper method to get observation history from the initial observation, by
        repeating it.

        Returns:
            obs_history (dict): a deque for each observation key, with an extra
                leading dimension of 1 for each key (for easy concatenation later)
        """
        obs_history = {}
        for k in init_obs:
            # Convert numpy array to torch tensor and move it to GPU
            obs_history[k] = deque(
                [torch.tensor(init_obs[k][None], device=self.device) for _ in range(10)], 
                maxlen=10,
            )
        return obs_history

    def update_obs(self, obs, action=None, reset=False):
        obs["timesteps"] = torch.tensor([self.timestep], device=self.device)
        
        if reset:
            obs["actions"] = torch.zeros(14*self.traj_len, device=self.device)
        else:
            self.timestep += 1
            obs["actions"] = torch.tensor(action[: 14*self.traj_len], device=self.device)

    def _get_stacked_obs_from_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a 
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        return {k: torch.cat(list(self.obs_history[k]), dim=0).to(self.device) for k in self.obs_history}

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        # self.sim_params.physx.max_gpu_contact_pairs = self.sim_params.physx.max_gpu_contact_pairs

        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        self.create_object_asset_dict(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        )
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_object_asset_dict(self, asset_root):
        self.object_asset_dict = {}
        print("ENTER ASSET CREATING!")
        for used_objects in self.used_training_objects:
            object_asset_file = self.asset_files_dict[used_objects]
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 1000
            object_asset_options.fix_base_link = True
            object_asset_options.flip_visual_attachments = False
            object_asset_options.collapse_fixed_joints = True
            if self.used_hand_type == "mano_free":
                object_asset_options.disable_gravity = False
            else:
                object_asset_options.disable_gravity = True
            object_asset_options.thickness = 0.001
            object_asset_options.angular_damping = 0.01
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 100000
            
            self.object_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            )

            object_asset_file = self.asset_files_dict[used_objects]
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 2000
            object_asset_options.disable_gravity = True
            object_asset_options.fix_base_link = True

            goal_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            )

            predict_goal_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            )

            self.object_asset_dict[used_objects] = {
                'obj': self.object_asset,
                'goal': goal_asset,
                'predict goal': predict_goal_asset,
            }

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"
        allegro_hand_asset_file = (
            "urdf/franka_description/robots/franka_panda_allegro_free_right.urdf"
        )
        allegro_hand_another_asset_file = (
            "urdf/franka_description/robots/franka_panda_allegro_free_left.urdf"
        )

        allegro_hand_asset_file = "urdf/shadow_hand_description/shadowhand_free.urdf"
        allegro_hand_another_asset_file = "urdf/shadow_hand_description/shadowhand_left_free.urdf"
        # object_asset_file = self.asset_files_dict["ball"]
        if "mano" in self.used_hand_type:
            allegro_hand_asset_file = "urdf/mano-urdf/urdf/mano_right_fixed.urdf"
            allegro_hand_another_asset_file = "urdf/mano-urdf/urdf/mano_left_fixed.urdf"
        if self.used_hand_type == "mano_free":
            allegro_hand_asset_file = "urdf/mano-urdf/urdf/mano_right_fixed_free.urdf"
            allegro_hand_another_asset_file = "urdf/mano-urdf/urdf/mano_left_fixed_free.urdf"
            
        if self.used_hand_type == "mano_shadow":
            allegro_hand_asset_file = "urdf/dex-urdf/robots/hands/shadow_hand/shadow_hand_right_for_ik.urdf"
            allegro_hand_another_asset_file = "urdf/dex-urdf/robots/hands/shadow_hand/shadow_hand_left_for_ik.urdf"
            
        self.use_joint_space_retargeting = False
        if self.use_joint_space_retargeting:
            allegro_hand_asset_file = "urdf/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf"
            allegro_hand_another_asset_file = "urdf/dex-urdf/robots/hands/shadow_hand/shadow_hand_left.urdf"

        # allegro_hand_asset_file = "urdf/franka_description/robots/franka_panda_allegro.urdf"
        # allegro_hand_another_asset_file = "urdf/franka_description/robots/franka_panda_allegro.urdf"
        self.object_name = self.used_training_objects[0]

        self.table_texture_files = (
            "../assets/arctic_assets/object_vtemplates/{}/material.jpg".format(self.object_name)
        )
        self.table_texture_handle = self.gym.create_texture_from_file(
            self.sim, self.table_texture_files
        )

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        # asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        # asset_options.override_com = True
        # asset_options.override_inertia = True
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params = gymapi.VhacdParams()
        # asset_options.vhacd_params.resolution = 3000000
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        allegro_hand_asset = self.gym.load_asset(
            self.sim, asset_root, allegro_hand_asset_file, asset_options
        )
        allegro_hand_another_asset = self.gym.load_asset(
            self.sim, asset_root, allegro_hand_another_asset_file, asset_options
        )

        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_actuators = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_tendons = self.gym.get_asset_tendon_count(allegro_hand_asset)

        print("self.num_allegro_hand_bodies: ", self.num_allegro_hand_bodies)
        print("self.num_allegro_hand_shapes: ", self.num_allegro_hand_shapes)
        print("self.num_allegro_hand_dofs: ", self.num_allegro_hand_dofs)
        print("self.num_allegro_hand_actuators: ", self.num_allegro_hand_actuators)
        print("self.num_allegro_hand_tendons: ", self.num_allegro_hand_tendons)

        self.actuated_dof_indices = [i for i in range(16)]

        # set allegro_hand dof properties
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)
        allegro_hand_another_dof_props = self.gym.get_asset_dof_properties(
            allegro_hand_another_asset
        )

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []
        self.a_allegro_hand_dof_lower_limits = []
        self.a_allegro_hand_dof_upper_limits = []
        self.allegro_hand_dof_default_pos = []
        self.allegro_hand_dof_default_vel = []
        self.allegro_hand_dof_stiffness = []
        self.allegro_hand_dof_damping = []
        self.allegro_hand_dof_effort = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            self.a_allegro_hand_dof_lower_limits.append(allegro_hand_another_dof_props['lower'][i])
            self.a_allegro_hand_dof_upper_limits.append(allegro_hand_another_dof_props['upper'][i])
            self.allegro_hand_dof_default_pos.append(0.0)
            self.allegro_hand_dof_default_vel.append(0.0)

            allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_NONE
            allegro_hand_another_dof_props['driveMode'][i] = gymapi.DOF_MODE_NONE
            if i < 6:
                allegro_hand_dof_props['stiffness'][i] = 400
                allegro_hand_dof_props['effort'][i] = 300
                allegro_hand_dof_props['damping'][i] = 80
                allegro_hand_another_dof_props['stiffness'][i] = 400
                allegro_hand_another_dof_props['effort'][i] = 300
                allegro_hand_another_dof_props['damping'][i] = 80

            else:
                allegro_hand_dof_props['velocity'][i] = 3.0
                allegro_hand_dof_props['stiffness'][i] = 30
                allegro_hand_dof_props['effort'][i] = 5
                allegro_hand_dof_props['damping'][i] = 1
                allegro_hand_another_dof_props['velocity'][i] = 3.0
                allegro_hand_another_dof_props['stiffness'][i] = 30
                allegro_hand_another_dof_props['effort'][i] = 5
                allegro_hand_another_dof_props['damping'][i] = 1

        self.actuated_dof_indices = to_torch(
            self.actuated_dof_indices, dtype=torch.long, device=self.device
        )
        self.allegro_hand_dof_lower_limits = to_torch(
            self.allegro_hand_dof_lower_limits, device=self.device
        )
        self.allegro_hand_dof_upper_limits = to_torch(
            self.allegro_hand_dof_upper_limits, device=self.device
        )
        self.a_allegro_hand_dof_lower_limits = to_torch(
            self.a_allegro_hand_dof_lower_limits, device=self.device
        )
        self.a_allegro_hand_dof_upper_limits = to_torch(
            self.a_allegro_hand_dof_upper_limits, device=self.device
        )
        self.allegro_hand_dof_default_pos = to_torch(
            self.allegro_hand_dof_default_pos, device=self.device
        )
        self.allegro_hand_dof_default_vel = to_torch(
            self.allegro_hand_dof_default_vel, device=self.device
        )

        home = os.path.expanduser('~')

        self.seq_list = []
        self.texture_list = []
        seq_list = ["01"]
        # seq_list = ["01"]
        num_total_traj = len(seq_list) * 4 * len(self.used_training_objects)
        traj_count = 0

        self.object_valid_list = []
        self.functional = "use"
        self.object_name = self.used_training_objects[0]

        if self.traj_index == "all":
            self.functional = ["use", "grab"]
            used_seq_list = ["01", "02", "04", "07", "08", "09"]
            used_sub_seq_list = ["01", "02", "03", "04"]
        else:
            self.functional = ["use"]
            if self.traj_index.split("_")[-1] == "grab":
                self.functional = ["grab"]
                
            used_seq_list = [self.traj_index.split("_")[0]] 
            used_sub_seq_list = [self.traj_index.split("_")[1]] 

        self.seq_list = []
        self.texture_list = []
        for functional in self.functional:
            for seq_i in used_seq_list:
                for j in used_sub_seq_list:
                    traj_count += 1
                    progress = int(traj_count / num_total_traj * 100)
                    print("\r", end="")
                    print(
                        "Process progress: {}%: ".format(progress),
                        "▋" * (progress // 2),
                        end="",
                    )

                    table_texture_files = (
                        "../assets/arctic_assets/object_vtemplates/{}/material.jpg".format(
                            self.object_name
                        )
                    )
                    table_texture_handle = self.gym.create_texture_from_file(
                        self.sim, table_texture_files
                    )
                    self.texture_list.append(table_texture_handle)

                    mano_p = "{}/arctic_yuanpei/data/arctic_data/data/raw_seqs/s{}/{}_{}_{}.mano.npy".format(
                        home, seq_i, self.object_name, functional, j
                    )
                    obj_p = "{}/arctic_yuanpei/data/arctic_data/data/raw_seqs/s{}/{}_{}_{}.object.npy".format(
                        home, seq_i, self.object_name, functional, j
                    )
                    # MANO
                    try:
                        data = np.load(
                            mano_p,
                            allow_pickle=True,
                        ).item()
                    except:
                        continue

                    mano_processed = (
                        "{}/arctic_yuanpei/outputs/processed/seqs/s{}/{}_{}_{}.npy".format(
                            home, seq_i, self.object_name, functional, j
                        )
                    )
                    self.mano_processed_data = np.load(
                        mano_processed,
                        allow_pickle=True,
                    ).item()

                    num_frames = len(data["right"]["rot"])

                    view_idx = 1

                    # view 1
                    cam2world_matrix = (
                        torch.tensor(
                            [
                                [0.8946, -0.4464, 0.0197, 0.1542],
                                [-0.1109, -0.2646, -0.9580, 0.9951],
                                [0.4328, 0.8548, -0.2862, 4.6415],
                                [0.0000, 0.0000, 0.0000, 1.0000],
                            ]
                        )[:3, :3]
                        .inverse()
                        .repeat(self.mano_processed_data["cam_coord"]["obj_rot_cam"].shape[0], 1, 1)
                    )

                    world2cam_matrix = torch.tensor(
                        [
                            [0.8946, -0.4464, 0.0197, 0.1542],
                            [-0.1109, -0.2646, -0.9580, 0.9951],
                            [0.4328, 0.8548, -0.2862, 4.6415],
                            [0.0000, 0.0000, 0.0000, 1.0000],
                        ]
                    )

                    # view 4
                    # world2cam_matrix = torch.tensor([[ 0.9194, -0.3786,  0.1069, -0.0453],
                    # [-0.2324, -0.7419, -0.6289,  0.8583],
                    # [ 0.3174,  0.5534, -0.7701,  5.0870],
                    # [ 0.0000,  0.0000,  0.0000,  1.0000]])

                    left_fingertip = self.mano_processed_data["world_coord"]["joints.left"][:, 16:21]
                    right_fingertip = self.mano_processed_data["world_coord"]["joints.right"][:, 16:21]
                    left_middle_finger = self.mano_processed_data["world_coord"]["joints.left"][:, [2,5,8,11,14]]
                    right_middle_finger = self.mano_processed_data["world_coord"]["joints.right"][:, [2,5,8,11,14]]
                    
                    import utils.rot as rot

                    quat_cam2world = rot.matrix_to_quaternion(cam2world_matrix).cuda()
                    obj_r_cam = rot.axis_angle_to_quaternion(
                        torch.FloatTensor(
                            self.mano_processed_data["cam_coord"]["obj_rot_cam"][:, view_idx, :]
                        ).cuda()
                    )
                    obj_r_world = rot.quaternion_to_axis_angle(
                        rot.quaternion_multiply(quat_cam2world, obj_r_cam)
                    )

                    rot_r_cam = rot.axis_angle_to_quaternion(
                        torch.FloatTensor(
                            self.mano_processed_data["cam_coord"]["rot_r_cam"][:, view_idx, :]
                        ).cuda()
                    )
                    rot_r_world = rot.quaternion_to_axis_angle(
                        rot.quaternion_multiply(quat_cam2world, rot_r_cam)
                    )
                    rot_l_cam = rot.axis_angle_to_quaternion(
                        torch.FloatTensor(
                            self.mano_processed_data["cam_coord"]["rot_l_cam"][:, view_idx, :]
                        ).cuda()
                    )
                    rot_l_world = rot.quaternion_to_axis_angle(
                        rot.quaternion_multiply(quat_cam2world, rot_l_cam)
                    )

                    obj_rot_quat = rot.axis_angle_to_quaternion(obj_r_world)
                    rot_r_quat = rot.axis_angle_to_quaternion(rot_r_world)
                    rot_l_quat = rot.axis_angle_to_quaternion(rot_l_world)

                    rot_r = torch.FloatTensor(data["right"]["rot"])
                    pose_r = torch.FloatTensor(data["right"]["pose"])
                    trans_r = torch.FloatTensor(data["right"]["trans"])
                    shape_r = torch.FloatTensor(data["right"]["shape"]).repeat(num_frames, 1)
                    fitting_err_r = data["right"]["fitting_err"]

                    rot_l = torch.FloatTensor(data["left"]["rot"])
                    pose_l = torch.FloatTensor(data["left"]["pose"])
                    trans_l = torch.FloatTensor(data["left"]["trans"])
                    shape_l = torch.FloatTensor(data["left"]["shape"]).repeat(num_frames, 1)
                    obj_params = torch.FloatTensor(np.load(obj_p, allow_pickle=True))
                    obj_params[:, 4:7] /= 1000
                    obj_params[:, 1:4] = obj_r_world
                    rot_r = rot_r_world
                    rot_l = rot_l_world

                    self.begin_frame = 30
                    # self.begin_frame = 180

                    self.rot_r = to_torch(rot_r, device=self.device)[self.begin_frame :]
                    self.trans_r = to_torch(trans_r, device=self.device)[self.begin_frame :]
                    self.rot_l = to_torch(rot_l, device=self.device)[self.begin_frame :]
                    self.trans_l = to_torch(trans_l, device=self.device)[self.begin_frame :]
                    self.obj_params = to_torch(obj_params, device=self.device)[self.begin_frame :]
                    
                    if "mano" in self.used_hand_type:
                        self.trans_l = to_torch(self.mano_processed_data["world_coord"]["joints.left"][:, 0], device=self.device)[self.begin_frame :]
                        self.trans_r = to_torch(self.mano_processed_data["world_coord"]["joints.right"][:, 0], device=self.device)[self.begin_frame :]

                    self.left_fingertip = to_torch(left_fingertip, device=self.device).view(left_fingertip.shape[0], 15)[self.begin_frame :]
                    self.right_fingertip = to_torch(right_fingertip, device=self.device).view(right_fingertip.shape[0], 15)[self.begin_frame :]
                    
                    self.left_middle_finger = to_torch(left_middle_finger, device=self.device).contiguous().view(left_middle_finger.shape[0], 15)[self.begin_frame :]
                    self.right_middle_finger = to_torch(right_middle_finger, device=self.device).contiguous().view(right_middle_finger.shape[0], 15)[self.begin_frame :]

                    self.obj_rot_quat = to_torch(obj_rot_quat, device=self.device)[
                        self.begin_frame :
                    ]
                    self.rot_r_quat = to_torch(rot_r_quat, device=self.device)[self.begin_frame :]
                    self.rot_l_quat = to_torch(rot_l_quat, device=self.device)[self.begin_frame :]

                    self.obj_rot_quat_tem = self.obj_rot_quat.clone()
                    self.obj_rot_quat[:, 0] = self.obj_rot_quat_tem[:, 1].clone()
                    self.obj_rot_quat[:, 1] = self.obj_rot_quat_tem[:, 2].clone()
                    self.obj_rot_quat[:, 2] = self.obj_rot_quat_tem[:, 3].clone()
                    self.obj_rot_quat[:, 3] = self.obj_rot_quat_tem[:, 0].clone()

                    self.rot_r_quat_tem = self.rot_r_quat.clone()
                    self.rot_r_quat[:, 0] = self.rot_r_quat_tem[:, 1].clone()
                    self.rot_r_quat[:, 1] = self.rot_r_quat_tem[:, 2].clone()
                    self.rot_r_quat[:, 2] = self.rot_r_quat_tem[:, 3].clone()
                    self.rot_r_quat[:, 3] = self.rot_r_quat_tem[:, 0].clone()

                    self.rot_l_quat_tem = self.rot_l_quat.clone()
                    self.rot_l_quat[:, 0] = self.rot_l_quat_tem[:, 1].clone()
                    self.rot_l_quat[:, 1] = self.rot_l_quat_tem[:, 2].clone()
                    self.rot_l_quat[:, 2] = self.rot_l_quat_tem[:, 3].clone()
                    self.rot_l_quat[:, 3] = self.rot_l_quat_tem[:, 0].clone()

                    # transform quat for arm
                    if "mano" in self.used_hand_type:
                        # self.left_transform_quat = to_torch(
                        #     [1.0, 0.0, 0, 0], dtype=torch.float
                        # ).repeat((self.rot_l_quat.shape[0], 1))
                        # self.rot_l_quat = quat_mul(self.rot_l_quat, self.left_transform_quat)
                        print(1)

                    else:
                        right_transform_quat = to_torch(
                            [0.0, -0.707, 0.0, 0.707], dtype=torch.float, device=self.device
                        ).repeat((self.rot_r_quat.shape[0], 1))
                        left_transform_quat = to_torch(
                            [0.707, 0.0, 0.707, 0.0], dtype=torch.float, device=self.device
                        ).repeat((self.rot_l_quat.shape[0], 1))
                        self.rot_l_quat = quat_mul(self.rot_l_quat, left_transform_quat)
                        self.rot_r_quat = quat_mul(self.rot_r_quat, right_transform_quat)

                    interpolate_time = 1
                    
                    self.rot_r = interpolate_tensor(self.rot_r, interpolate_time)
                    self.trans_r = interpolate_tensor(self.trans_r, interpolate_time)
                    self.rot_l = interpolate_tensor(self.rot_l, interpolate_time)
                    self.trans_l = interpolate_tensor(self.trans_l, interpolate_time)
                    self.obj_params = interpolate_tensor(self.obj_params, interpolate_time)
                    self.obj_rot_quat = interpolate_tensor(self.obj_rot_quat, interpolate_time)
                    self.rot_r_quat = interpolate_tensor(self.rot_r_quat, interpolate_time)
                    self.rot_l_quat = interpolate_tensor(self.rot_l_quat, interpolate_time)
                    self.left_fingertip = interpolate_tensor(self.left_fingertip, interpolate_time)
                    self.right_fingertip = interpolate_tensor(self.right_fingertip, interpolate_time)
                    self.left_middle_finger = interpolate_tensor(self.left_middle_finger, interpolate_time)
                    self.right_middle_finger = interpolate_tensor(self.right_middle_finger, interpolate_time)
                    
                    for i, rot_quat in enumerate(self.obj_rot_quat):
                        if i > 0:
                            if calculate_frobenius_norm_of_rotation_difference(rot_quat, last_obj_rot_global, device=self.device) > 0.5:
                                self.obj_rot_quat[i] = last_obj_rot_global.clone()
                            
                        last_obj_rot_global = rot_quat.clone()

                    # offset
                    if not "mano" in self.used_hand_type:
                        self.trans_r[:, 2] += -0.07
                        self.trans_r[:, 0] += 0.07
                        self.trans_l[:, 2] += -0.05
                        self.trans_l[:, 0] += -0.07

                        self.trans_r[:, 1] += 0.04
                        self.trans_l[:, 1] += 0.04

                    self.seq_list.append(
                        {
                            "rot_r": self.rot_r.clone(),
                            "trans_r": self.trans_r.clone(),
                            "rot_l": self.rot_l.clone(),
                            "trans_l": self.trans_l.clone(),
                            "obj_params": self.obj_params.clone(),
                            "obj_rot_quat": self.obj_rot_quat.clone(),
                            "rot_r_quat": self.rot_r_quat.clone(),
                            "rot_l_quat": self.rot_l_quat.clone(),
                            "left_fingertip": self.left_fingertip.clone(),
                            "right_fingertip": self.right_fingertip.clone(),
                            "left_middle_finger": self.left_middle_finger.clone(),
                            "right_middle_finger": self.right_middle_finger.clone(),
                        }
                    )

        print("seq_num: ", len(self.seq_list))
        self.seq_list_i = [i for i in range(len(self.seq_list))]

        self.arctic_traj_len = 1000
        # self.arctic_traj_len = 200

        self.rot_r = torch.zeros((self.num_envs, self.arctic_traj_len, 3), device=self.device, dtype=torch.float)
        self.trans_r = torch.zeros((self.num_envs, self.arctic_traj_len, 3), device=self.device, dtype=torch.float)
        self.rot_l = torch.zeros((self.num_envs, self.arctic_traj_len, 3), device=self.device, dtype=torch.float)
        self.trans_l = torch.zeros((self.num_envs, self.arctic_traj_len, 3), device=self.device, dtype=torch.float)
        self.obj_params = torch.zeros(
            (self.num_envs, self.arctic_traj_len, 7), device=self.device, dtype=torch.float
        )
        self.obj_rot_quat = torch.zeros(
            (self.num_envs, self.arctic_traj_len, 4), device=self.device, dtype=torch.float
        )
        self.rot_r_quat = torch.zeros(
            (self.num_envs, self.arctic_traj_len, 4), device=self.device, dtype=torch.float
        )
        self.rot_l_quat = torch.zeros(
            (self.num_envs, self.arctic_traj_len, 4), device=self.device, dtype=torch.float
        )
        self.left_fingertip = torch.zeros(
            (self.num_envs, self.arctic_traj_len, 15), device=self.device, dtype=torch.float
        )
        self.right_fingertip = torch.zeros(
            (self.num_envs, self.arctic_traj_len, 15), device=self.device, dtype=torch.float
        )
        self.left_middle_finger = torch.zeros(
            (self.num_envs, self.arctic_traj_len, 15), device=self.device, dtype=torch.float
        )
        self.right_middle_finger = torch.zeros(
            (self.num_envs, self.arctic_traj_len, 15), device=self.device, dtype=torch.float
        )
        
        for i in range(self.num_envs):
            seq_idx = i % len(self.seq_list)
            self.seq_idx_tensor = to_torch([range(self.num_envs)], dtype=int, device=self.device)
            self.rot_r[i] = self.seq_list[seq_idx]["rot_r"][:self.arctic_traj_len].clone()
            self.trans_r[i] = self.seq_list[seq_idx]["trans_r"][:self.arctic_traj_len].clone()
            self.rot_l[i] = self.seq_list[seq_idx]["rot_l"][:self.arctic_traj_len].clone()
            self.trans_l[i] = self.seq_list[seq_idx]["trans_l"][:self.arctic_traj_len].clone()
            self.obj_params[i] = self.seq_list[seq_idx]["obj_params"][:self.arctic_traj_len].clone()
            self.obj_rot_quat[i] = self.seq_list[seq_idx]["obj_rot_quat"][:self.arctic_traj_len].clone()
            self.rot_r_quat[i] = self.seq_list[seq_idx]["rot_r_quat"][:self.arctic_traj_len].clone()
            self.rot_l_quat[i] = self.seq_list[seq_idx]["rot_l_quat"][:self.arctic_traj_len].clone()
            self.left_fingertip[i] = self.seq_list[seq_idx]["left_fingertip"][:self.arctic_traj_len].clone()
            self.right_fingertip[i] = self.seq_list[seq_idx]["right_fingertip"][:self.arctic_traj_len].clone()
            self.left_middle_finger[i] = self.seq_list[seq_idx]["left_middle_finger"][:self.arctic_traj_len].clone()
            self.right_middle_finger[i] = self.seq_list[seq_idx]["right_middle_finger"][:self.arctic_traj_len].clone()
            
        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.disable_gravity = True

        self.object_radius = 0.06
        object_asset = self.gym.create_sphere(self.sim, 0.12, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.create_sphere(self.sim, 0.04, object_asset_options)

        # allegro_hand_start_pose = gymapi.Transform()
        # allegro_hand_start_pose.p = gymapi.Vec3(-self.obj_params[1, 4] - 0.84, -self.obj_params[1, 5] - 0.64, -self.obj_params[1, 6] + 1.5)
        # allegro_hand_start_pose.r = gymapi.Quat().from_euler_zyx(-self.obj_params[1, 1] - 1.571, -self.obj_params[1, 2] - 1.571, -self.obj_params[1, 3])

        # allegro_another_hand_start_pose = gymapi.Transform()
        # allegro_another_hand_start_pose.p = gymapi.Vec3(-self.obj_params[1, 4] - 0.84, -self.obj_params[1, 5] - 0.64, -self.obj_params[1, 6] + 1.5)
        # allegro_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(-self.obj_params[1, 1] - 1.571, -self.obj_params[1, 2] - 1.571, -self.obj_params[1, 3])

        # object_start_pose = gymapi.Transform()
        # object_start_pose.p = gymapi.Vec3(-self.obj_params[1, 4] - 0.84, -self.obj_params[1, 5] - 0.64, -self.obj_params[1, 6] + 1.5)
        # object_start_pose.r = gymapi.Quat().from_euler_zyx(-self.obj_params[1, 1] - 1.571, -self.obj_params[1, 2] - 1.571, -self.obj_params[1, 3])
        # object_start_pose.r = gymapi.Quat().from_euler_zyx(-self.obj_params[1, 1] - 1.571, -self.obj_params[1, 2] - 1.571, -self.obj_params[1, 3])

        allegro_hand_start_pose = gymapi.Transform()
        allegro_hand_start_pose.p = gymapi.Vec3(0.0, 0, 0.0)
        allegro_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        allegro_another_hand_start_pose = gymapi.Transform()
        allegro_another_hand_start_pose.p = gymapi.Vec3(-0.0, 0, 0.0)
        allegro_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0, -0.25, 0)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        self.goal_displacement = gymapi.Vec3(-0.0, 0.0, 0.0)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z],
            device=self.device,
        )
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.0

        # create table asset
        table_dims = gymapi.Vec3(0.7, 1.0, 0.76)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001

        table_asset = self.gym.create_box(
            self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options
        )
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0, 0, 0.0)

        # create support box asset
        support_box_dims = gymapi.Vec3(0.15, 0.15, 0.20)
        support_box_asset_options = gymapi.AssetOptions()
        support_box_asset_options.fix_base_link = True
        support_box_asset_options.flip_visual_attachments = True
        support_box_asset_options.collapse_fixed_joints = True
        support_box_asset_options.disable_gravity = True
        support_box_asset_options.thickness = 0.001

        support_box_asset = self.gym.create_box(
            self.sim,
            support_box_dims.x,
            support_box_dims.y,
            support_box_dims.z,
            support_box_asset_options,
        )
        support_box_pose = gymapi.Transform()
        support_box_pose.p = gymapi.Vec3(0.0, -0.10, 0.5 * (2 * table_dims.z + support_box_dims.z))
        support_box_pose.r = gymapi.Quat().from_euler_zyx(-0, 0, 0.0)

        # compute aggregate size
        max_agg_bodies = self.num_allegro_hand_bodies * 2 + 2 + 50
        max_agg_shapes = self.num_allegro_hand_shapes * 2 + 2 + 50

        self.allegro_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []
        self.another_hand_start_states = []

        self.hand_indices = []
        self.another_hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.predict_goal_object_indices = []

        self.table_indices = []
        self.support_box_indices = []

        if self.enable_camera_sensors:
            self.cameras = []
            self.camera_tensors = []
            self.camera_view_matrixs = []
            self.camera_proj_matrixs = []

            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 1920
            self.camera_props.height = 1080
            self.camera_props.enable_tensors = True

            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            self.pointCloudDownsampleNum = 384
            self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)
            self.point_clouds = torch.zeros(
                (self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device
            )

            self.camera_v2, self.camera_u2 = torch.meshgrid(
                self.camera_v, self.camera_u, indexing='ij'
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            
            self.video_out_list = []
            for i in range(len(self.seq_list)):
                save_video_dir = '/home/user/DexterousHandEnvs/dexteroushandenvs/videos/fingertip_ik/'
                if not os.path.exists(save_video_dir):
                    os.makedirs(save_video_dir)
                    
                self.out = cv2.VideoWriter(save_video_dir + '{}_{}_{}_reference.mp4'.format(self.used_hand_type, self.used_training_objects[0], i), fourcc, 30.0, (1920, 1080))
                self.video_out_list.append(self.out)
                
            if self.point_cloud_debug:
                import open3d as o3d
                from utils.o3dviewer import PointcloudVisualizer

                self.pointCloudVisualizer = PointcloudVisualizer()
                self.pointCloudVisualizerInitialized = False
                self.o3d_pc = o3d.geometry.PointCloud()
            else:
                self.pointCloudVisualizer = None

        import open3d as o3d

        self.origin_point_clouds = torch.zeros((self.num_envs, 10000, 3), device=self.device)
        self.pointCloudDownsampleNum = 384
        self.point_clouds = torch.zeros(
            (self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device
        )

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            if self.used_hand_type == "mano_free":
                c = 0
            else:
                c = 4
            allegro_hand_actor = self.gym.create_actor(
                env_ptr,
                allegro_hand_asset,
                allegro_hand_start_pose,
                "hand",
                i + 4 * self.num_envs,
                0,
                0,
            )
            allegro_hand_another_actor = self.gym.create_actor(
                env_ptr,
                allegro_hand_another_asset,
                allegro_another_hand_start_pose,
                "another_hand",
                i + 5 * self.num_envs,
                0,
                0,
            )

            self.hand_start_states.append(
                [
                    allegro_hand_start_pose.p.x,
                    allegro_hand_start_pose.p.y,
                    allegro_hand_start_pose.p.z,
                    allegro_hand_start_pose.r.x,
                    allegro_hand_start_pose.r.y,
                    allegro_hand_start_pose.r.z,
                    allegro_hand_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )

            self.another_hand_start_states.append(
                [
                    allegro_another_hand_start_pose.p.x,
                    allegro_another_hand_start_pose.p.y,
                    allegro_another_hand_start_pose.p.z,
                    allegro_another_hand_start_pose.r.x,
                    allegro_another_hand_start_pose.r.y,
                    allegro_another_hand_start_pose.r.z,
                    allegro_another_hand_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )

            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            self.gym.set_actor_dof_properties(
                env_ptr, allegro_hand_another_actor, allegro_hand_another_dof_props
            )
            another_hand_idx = self.gym.get_actor_index(
                env_ptr, allegro_hand_another_actor, gymapi.DOMAIN_SIM
            )
            self.another_hand_indices.append(another_hand_idx)

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, allegro_hand_actor)
            hand_rigid_body_index = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]

            # add object
            index = i % len(self.used_training_objects)
            select_obj = self.used_training_objects[index]
            
            if self.used_hand_type == "mano_free":
                c = 0
            else:
                c = 3
                
            object_handle = self.gym.create_actor(
                env_ptr,
                self.object_asset_dict[select_obj]['obj'],
                object_start_pose,
                "object",
                i + c*self.num_envs,
                0,
                0,
            )

            # object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append(
                [
                    object_start_pose.p.x,
                    object_start_pose.p.y,
                    object_start_pose.p.z,
                    object_start_pose.r.x,
                    object_start_pose.r.y,
                    object_start_pose.r.z,
                    object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.gym.set_rigid_body_texture(
                env_ptr, object_handle, 0, gymapi.MESH_VISUAL, self.texture_list[index]
            )
            self.gym.set_rigid_body_texture(
                env_ptr, object_handle, 1, gymapi.MESH_VISUAL, self.texture_list[index]
            )
            self.object_indices.append(object_idx)

            lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            for lego_body_prop in lego_body_props:
                lego_body_prop.mass *= 1
            self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, lego_body_props)

            # object_dof_props = self.gym.get_actor_dof_properties(env_ptr, object_handle)
            # for object_dof_prop in object_dof_props:
            #     object_dof_prop[4] = 100
            #     object_dof_prop[5] = 50
            #     object_dof_prop[6] = 5
            #     object_dof_prop[7] = 1
            # self.gym.set_actor_dof_properties(env_ptr, object_handle, object_dof_props)

            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            for object_shape_prop in object_shape_props:
                object_shape_prop.restitution = 0
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)
            
            # print(self.gym.get_actor_dof_dict(env_ptr, allegro_hand_actor))
            # exit()

            hand_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, allegro_hand_actor
            )
            for hand_shape_prop in hand_shape_props:
                hand_shape_prop.restitution = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, hand_shape_props)

            # generate offline point cloud
            # pcd = o3d.io.read_triangle_mesh(self.asset_point_cloud_files_dict[select_obj])
            # self.origin_point_clouds[i] = torch.tensor([pcd.vertices], dtype=torch.float, device=self.device)

            # add goal object
            goal_handle = self.gym.create_actor(
                env_ptr,
                self.object_asset_dict[select_obj]['goal'],
                goal_start_pose,
                "goal_object",
                i + self.num_envs,
                0,
                0,
            )
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            # add goal object
            # predict_goal_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['predict goal'], goal_start_pose, "predict_goal_object", i + self.num_envs * 2, 0, 0)
            # predict_goal_object_idx = self.gym.get_actor_index(env_ptr, predict_goal_handle, gymapi.DOMAIN_SIM)
            # self.predict_goal_object_indices.append(predict_goal_object_idx)
            # self.gym.set_rigid_body_color(env_ptr, predict_goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.4, 0.))

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.9, 0.8)
            )
            self.table_indices.append(table_idx)

            # add support box
            support_box_handle = self.gym.create_actor(
                env_ptr, support_box_asset, support_box_pose, "support_box", i, 0, 0
            )
            support_box_idx = self.gym.get_actor_index(
                env_ptr, support_box_handle, gymapi.DOMAIN_SIM
            )
            self.gym.set_rigid_body_color(
                env_ptr, support_box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.9, 0.8)
            )
            self.support_box_indices.append(support_box_idx)

            if self.enable_camera_sensors:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                origin = self.gym.get_env_origin(env_ptr)
                self.gym.set_camera_location(
                    camera_handle, env_ptr, gymapi.Vec3(0.0, 0.5, 1.6), gymapi.Vec3(0, -0.5, 0.5)
                )
                
                camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR
                )
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                cam_vinv = torch.inverse(
                    (
                        torch.tensor(
                            self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)
                        )
                    )
                ).to(self.device)
                cam_proj = torch.tensor(
                    self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle),
                    device=self.device,
                )

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98)
                )
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98)
                )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_hand_actor)

            if self.enable_camera_sensors:
                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

        another_sensor_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, allegro_hand_another_actor, sensor_name)
            for sensor_name in self.contact_sensor_names
        ]

        sensor_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, allegro_hand_actor, sensor_name)
            for sensor_name in self.contact_sensor_names
        ]

        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)
        self.another_sensor_handle_indices = to_torch(another_sensor_handles, dtype=torch.int64)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(
            self.num_envs, 13
        )
        self.another_hand_start_states = to_torch(
            self.another_hand_start_states, device=self.device
        ).view(self.num_envs, 13)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices = to_torch(
            self.another_hand_indices, dtype=torch.long, device=self.device
        )

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(
            self.goal_object_indices, dtype=torch.long, device=self.device
        )
        self.predict_goal_object_indices = to_torch(
            self.predict_goal_object_indices, dtype=torch.long, device=self.device
        )

        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)
        self.support_box_indices = to_torch(
            self.support_box_indices, dtype=torch.long, device=self.device
        )

        self.total_steps = 0
        self.success_buf = torch.zeros_like(self.rew_buf)
        self.hit_success_buf = torch.zeros_like(self.rew_buf)

    def get_internal_state(self):
        return self.root_state_tensor[self.object_indices, 3:7]

    def get_internal_info(self, key):
        if key == 'target':
            return self.debug_target
        elif key == 'qpos':
            return self.debug_qpos
        elif key == 'contact':
            return self.finger_contacts
        return None
        # return {'target': self.debug_target, 'gt': self.debug_qpos}

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_hand_reward(
            self.rew_buf,
            self.reset_buf,
            self.reset_goal_buf,
            self.progress_buf,
            self.successes,
            self.consecutive_successes,
            self.left_contacts,
            self.right_contacts,
            self.allegro_left_hand_pos,
            self.allegro_right_hand_pos,
            self.allegro_left_hand_rot,
            self.allegro_right_hand_rot,
            self.max_episode_length,
            self.object_base_pos,
            self.object_base_rot,
            self.goal_pos,
            self.goal_rot,
            self.allegro_left_hand_dof,
            self.allegro_right_hand_dof,
            self.object_dof,
            self.trans_r,
            self.trans_l,
            self.rot_r_quat,
            self.rot_l_quat,
            self.obj_params,
            self.obj_rot_quat,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.rot_eps,
            self.actions,
            self.action_penalty_scale,
            self.a_hand_palm_pos,
            self.last_actions,
            self.success_tolerance,
            self.reach_goal_bonus,
            self.fall_dist,
            self.fall_penalty,
            self.end_step_buf,
            self.seq_idx_tensor,
            self.max_consecutive_successes,
            self.av_factor,
            (self.object_type == "pen"),
        )

        self.last_actions = self.actions.clone()

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        self.total_steps += 1

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print(
                "Direct average consecutive successes = {:.1f}".format(
                    direct_average_successes / (self.total_resets + self.num_envs)
                )
            )
            if self.total_resets > 0:
                print(
                    "Post-Reset average consecutive successes = {:.1f}".format(
                        self.total_successes / self.total_resets
                    )
                )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.allegro_right_hand_base_pos = self.root_state_tensor[self.hand_indices, 0:3]
        self.allegro_right_hand_base_rot = self.root_state_tensor[self.hand_indices, 3:7]

        self.allegro_left_hand_base_pos = self.root_state_tensor[self.another_hand_indices, 0:3]
        self.allegro_left_hand_base_rot = self.root_state_tensor[self.another_hand_indices, 3:7]

        self.object_base_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_base_rot = self.root_state_tensor[self.object_indices, 3:7]

        # self.allegro_right_hand_pos = self.root_state_tensor[self.hand_indices, 0:3]
        # self.allegro_right_hand_rot = self.root_state_tensor[self.hand_indices, 3:7]
        # self.allegro_right_hand_linvel = self.root_state_tensor[self.hand_indices, 7:10]
        # self.allegro_right_hand_angvel = self.root_state_tensor[self.hand_indices, 10:13]

        # self.allegro_left_hand_pos = self.root_state_tensor[self.another_hand_indices, 0:3]
        # self.allegro_left_hand_rot = self.root_state_tensor[self.another_hand_indices, 3:7]
        # self.allegro_left_hand_linvel = self.root_state_tensor[self.another_hand_indices, 7:10]
        # self.allegro_left_hand_angvel = self.root_state_tensor[self.another_hand_indices, 10:13]

        self.allegro_right_hand_pos = self.rigid_body_states[:, 6, 0:3]
        self.allegro_right_hand_rot = self.rigid_body_states[:, 6, 3:7]
        self.allegro_right_hand_linvel = self.rigid_body_states[:, 6, 7:10]
        self.allegro_right_hand_angvel = self.rigid_body_states[:, 6, 10:13]

        self.allegro_left_hand_pos = self.rigid_body_states[:, 6 + 25, 0:3]
        self.allegro_left_hand_rot = self.rigid_body_states[:, 6 + 25, 3:7]
        self.allegro_left_hand_linvel = self.rigid_body_states[:, 6 + 25, 7:10]
        self.allegro_left_hand_angvel = self.rigid_body_states[:, 6 + 25, 10:13]

        self.a_hand_palm_pos = self.allegro_left_hand_pos.clone()

        # self.allegro_left_hand_pos = self.allegro_left_hand_pos + quat_apply(self.allegro_left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.08)
        # self.allegro_left_hand_pos = self.allegro_left_hand_pos + quat_apply(self.allegro_left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.allegro_right_hand_dof = self.allegro_hand_dof_pos.clone()
        self.allegro_left_hand_dof = self.allegro_hand_another_dof_pos.clone()
        self.object_dof = self.object_dof_pos.clone()

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.leeft_hand_ee_rot = self.rigid_body_states[
            :, self.another_hand_base_rigid_body_index, 3:7
        ]

        contacts = self.contact_tensor.reshape(self.num_envs, -1, 3)  # 39+27
        self.right_contacts = contacts[:, self.sensor_handle_indices, :]  # 12
        self.right_contacts = torch.norm(self.right_contacts, dim=-1)
        self.right_contacts = torch.where(self.right_contacts >= 0.1, 1.0, 0.0)

        # for i in range(len(self.right_contacts[0])):
        #     if self.right_contacts[0][i] == 1.0:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.3, 0.3))
        #     else:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))

        self.left_contacts = contacts[:, self.another_sensor_handle_indices, :]  # 12
        self.left_contacts = torch.norm(self.left_contacts, dim=-1)
        self.left_contacts = torch.where(self.left_contacts >= 0.1, 1.0, 0.0)

        # for i in range(len(self.left_contacts[0])):
        #     if self.left_contacts[0][i] == 1.0:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.another_hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.3, 0.3))
        #     else:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.another_hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (self.num_envs, 63), device=self.device)

        self.aux_up_pos = to_torch([0, -0.52, 0.45], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        
        # self.compute_sim2real_observation(rand_floats)
        # # self.compute_full_state()

        # if self.asymmetric_obs:
        #     self.compute_sim2real_asymmetric_obs(rand_floats)

    def compute_sim2real_observation(self, rand_floats):
        # origin obs
        self.obs_buf[:, 0 + 6 : 18 + 6] = unscale(
            self.allegro_hand_dof_pos[:, 0 + 6 : 18 + 6],
            self.allegro_hand_dof_lower_limits[0 + 6 : 18 + 6],
            self.allegro_hand_dof_upper_limits[0 + 6 : 18 + 6],
        )
        self.obs_buf[:, 18 + 6 : 36 + 6] = unscale(
            self.allegro_hand_another_dof_pos[:, 0 + 6 : 18 + 6],
            self.allegro_hand_dof_lower_limits[0 + 6 : 18 + 6],
            self.allegro_hand_dof_upper_limits[0 + 6 : 18 + 6],
        )

        self.obs_buf[:, 36:84] = self.actions

        self.obs_buf[:, 96:99] = self.allegro_right_hand_pos
        self.obs_buf[:, 99:103] = self.allegro_right_hand_rot

        self.obs_buf[:, 103:106] = self.allegro_left_hand_pos
        self.obs_buf[:, 106:110] = self.allegro_left_hand_rot

        self.obs_buf[:, 110:117] = self.object_pose
        self.obs_buf[:, 117:118] = self.object_dof_pos
        # self.obs_buf[:, 118:119] = self.progress_buf.unsqueeze(-1)

        self.obs_buf[:, 123:126] = self.object_pos - self.obj_params[
            self.seq_idx_tensor, self.progress_buf, 4:7
        ].squeeze(0)
        self.obs_buf[:, 126:130] = quat_mul(
            self.object_rot,
            quat_conjugate(self.obj_rot_quat[self.seq_idx_tensor, self.progress_buf].squeeze(0)),
        )

        self.obs_buf[:, 130:131] = self.object_dof - self.obj_params[
            self.seq_idx_tensor, self.progress_buf, 0:1
        ].squeeze(0)

        self.stack_frame = 10
        for i in range(self.stack_frame):
            self.obs_buf[:, 144 + 22 * i : 147 + 22 * i] = self.obj_params[
                self.seq_idx_tensor, self.progress_buf + i, 4:7
            ].squeeze(0)
            self.obs_buf[:, 147 + 22 * i : 151 + 22 * i] = self.obj_rot_quat[
                self.seq_idx_tensor, self.progress_buf + i
            ].squeeze(0)
            self.obs_buf[:, 151 + 22 * i : 152 + 22 * i] = self.obj_params[
                self.seq_idx_tensor, self.progress_buf, 0:1
            ].squeeze(0)

            self.obs_buf[:, 152 + 22 * i : 155 + 22 * i] = self.trans_l[
                self.seq_idx_tensor, self.progress_buf + i
            ]
            self.obs_buf[:, 155 + 22 * i : 159 + 22 * i] = self.rot_l_quat[
                self.seq_idx_tensor, self.progress_buf + i
            ]
            self.obs_buf[:, 159 + 22 * i : 162 + 22 * i] = self.trans_r[
                self.seq_idx_tensor, self.progress_buf + i
            ]
            self.obs_buf[:, 162 + 22 * i : 166 + 22 * i] = self.rot_r_quat[
                self.seq_idx_tensor, self.progress_buf + i
            ]

    def compute_sim2real_asymmetric_obs(self, rand_floats):
        # visualize
        self.states_buf[:, 0:18] = unscale(
            self.allegro_hand_dof_pos[:, 0 + 6 : 18 + 6],
            self.allegro_hand_dof_lower_limits[0 + 6 : 18 + 6],
            self.allegro_hand_dof_upper_limits[0 + 6 : 18 + 6],
        )
        self.states_buf[:, 18:36] = unscale(
            self.allegro_hand_another_dof_pos[:, 0 + 6 : 18 + 6],
            self.allegro_hand_dof_lower_limits[0 + 6 : 18 + 6],
            self.allegro_hand_dof_upper_limits[0 + 6 : 18 + 6],
        )

        self.states_buf[:, 36:84] = self.actions

        self.states_buf[:, 84:87] = self.allegro_right_hand_linvel
        self.states_buf[:, 87:90] = self.allegro_right_hand_angvel
        self.states_buf[:, 90:93] = self.allegro_left_hand_linvel
        self.states_buf[:, 93:96] = self.allegro_left_hand_angvel

        self.states_buf[:, 96:99] = self.allegro_right_hand_pos
        self.states_buf[:, 99:103] = self.allegro_right_hand_rot

        self.states_buf[:, 103:106] = self.allegro_left_hand_pos
        self.states_buf[:, 106:110] = self.allegro_left_hand_rot

        self.states_buf[:, 110:117] = self.object_pose
        self.states_buf[:, 117:120] = self.object_linvel
        self.states_buf[:, 120:123] = self.object_angvel

        self.states_buf[:, 123:126] = self.object_pos - self.obj_params[
            self.seq_idx_tensor, self.progress_buf, 4:7
        ].squeeze(0)
        self.states_buf[:, 126:130] = quat_mul(
            self.object_rot,
            quat_conjugate(self.obj_rot_quat[self.seq_idx_tensor, self.progress_buf].squeeze(0)),
        )
        self.states_buf[:, 130:133] = self.allegro_left_hand_pos - self.trans_l[
            self.seq_idx_tensor, self.progress_buf
        ].squeeze(0)
        self.states_buf[:, 133:137] = quat_mul(
            self.allegro_left_hand_rot,
            quat_conjugate(self.rot_l_quat[self.seq_idx_tensor, self.progress_buf].squeeze(0)),
        )
        self.states_buf[:, 137:140] = self.allegro_right_hand_pos - self.trans_r[
            self.seq_idx_tensor, self.progress_buf
        ].squeeze(0)
        self.states_buf[:, 140:144] = quat_mul(
            self.allegro_right_hand_rot,
            quat_conjugate(self.rot_r_quat[self.seq_idx_tensor, self.progress_buf].squeeze(0)),
        )

        self.stack_frame = 10
        for i in range(3):
            self.states_buf[:, 144 + 22 * i : 147 + 22 * i] = self.obj_params[
                self.seq_idx_tensor, self.progress_buf + i, 4:7
            ].squeeze(0)
            self.states_buf[:, 147 + 22 * i : 151 + 22 * i] = self.obj_rot_quat[
                self.seq_idx_tensor, self.progress_buf + i
            ].squeeze(0)
            self.states_buf[:, 151 + 22 * i : 154 + 22 * i] = self.trans_l[
                self.seq_idx_tensor, self.progress_buf + i
            ].squeeze(0)
            self.states_buf[:, 154 + 22 * i : 158 + 22 * i] = self.rot_l_quat[
                self.seq_idx_tensor, self.progress_buf + i
            ].squeeze(0)
            self.states_buf[:, 158 + 22 * i : 161 + 22 * i] = self.trans_r[
                self.seq_idx_tensor, self.progress_buf + i
            ].squeeze(0)
            self.states_buf[:, 161 + 22 * i : 165 + 22 * i] = self.rot_r_quat[
                self.seq_idx_tensor, self.progress_buf + i
            ].squeeze(0)
            self.states_buf[:, 165 + 22 * i : 166 + 22 * i] = self.obj_params[
                self.seq_idx_tensor, self.progress_buf, 0:1
            ].squeeze(0)

        self.states_buf[:, 210:211] = self.object_dof - self.obj_params[
            self.seq_idx_tensor, self.progress_buf, 0:1
        ].squeeze(0)


    def reset(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.perturb_direction[env_ids] = torch_rand_float(
            -1, 1, (len(env_ids), 6), device=self.device
        ).squeeze(-1)

        # generate random values
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), self.num_allegro_hand_dofs * 2 + 5), device=self.device
        )

        # self.root_state_tensor[self.another_hand_indices[env_ids], 2] = -0.05 + rand_floats[:, 4] * 0.01

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[
            env_ids
        ].clone()
        self.root_state_tensor[self.hand_indices[env_ids]] = self.hand_start_states[env_ids].clone()
        self.root_state_tensor[self.another_hand_indices[env_ids]] = self.another_hand_start_states[
            env_ids
        ].clone()

        self.object_pose_for_open_loop[env_ids] = self.root_state_tensor[
            self.object_indices[env_ids], 0:7
        ].clone()

        object_indices = torch.unique(
            torch.cat(
                [
                    self.object_indices[env_ids],
                    self.goal_object_indices[env_ids],
                    self.table_indices[env_ids],
                    self.support_box_indices[env_ids],
                    self.goal_object_indices[goal_env_ids],
                ]
            ).to(torch.int32)
        )
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset shadow hand
        pos = self.allegro_hand_default_dof_pos
        another_pos = self.another_allegro_hand_default_dof_pos

        self.allegro_hand_dof_pos[env_ids, :] = pos
        self.allegro_hand_another_dof_pos[env_ids, :] = another_pos

        self.allegro_hand_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel

        self.allegro_hand_another_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel

        self.prev_targets[env_ids, : self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, : self.num_allegro_hand_dofs] = pos

        self.prev_targets[
            env_ids, self.num_allegro_hand_dofs : self.num_allegro_hand_dofs * 2
        ] = another_pos
        self.cur_targets[
            env_ids, self.num_allegro_hand_dofs : self.num_allegro_hand_dofs * 2
        ] = another_pos

        # reset object
        self.object_dof_pos[env_ids, :] = self.object_default_dof_pos
        self.object_dof_vel[env_ids, :] = torch.zeros_like(self.object_dof_vel[env_ids, :])
        self.prev_targets[env_ids, 2 * self.num_allegro_hand_dofs :] = self.object_default_dof_pos
        self.cur_targets[env_ids, 2 * self.num_allegro_hand_dofs :] = self.object_default_dof_pos

        if self.used_hand_type == "mano_free":
            self.prev_targets[env_ids, 0:3] = self.trans_r[env_ids, 0]
            self.prev_targets[env_ids, 3:6] = xyzw_quaternion_to_euler_xyz(self.rot_r_quat[env_ids, 0])
            self.cur_targets[env_ids, 0:3] = self.trans_r[env_ids, 0]
            self.cur_targets[env_ids, 3:6] = xyzw_quaternion_to_euler_xyz(self.rot_r_quat[env_ids, 0])
            
            self.prev_targets[env_ids, 0+self.num_allegro_hand_dofs:3+self.num_allegro_hand_dofs] = self.trans_l[env_ids, 0]
            self.prev_targets[env_ids, 3+self.num_allegro_hand_dofs:6+self.num_allegro_hand_dofs] = xyzw_quaternion_to_euler_xyz(self.rot_l_quat[env_ids, 0])
            self.cur_targets[env_ids, 0+self.num_allegro_hand_dofs:3+self.num_allegro_hand_dofs] = self.trans_l[env_ids, 0]
            self.cur_targets[env_ids, 3+self.num_allegro_hand_dofs:6+self.num_allegro_hand_dofs] = xyzw_quaternion_to_euler_xyz(self.rot_l_quat[env_ids, 0])
            
            self.allegro_hand_dof_pos[env_ids, 0:3] = self.trans_r[env_ids, 0]
            self.allegro_hand_dof_pos[env_ids, 3:6] = xyzw_quaternion_to_euler_xyz(self.rot_r_quat[env_ids, 0])

            self.allegro_hand_another_dof_pos[env_ids, 0:3] = self.trans_l[env_ids, 0]
            self.allegro_hand_another_dof_pos[env_ids, 3:6] = xyzw_quaternion_to_euler_xyz(self.rot_l_quat[env_ids, 0])

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        another_hand_indices = self.another_hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(
            torch.cat(
                [
                    hand_indices,
                    another_hand_indices,
                    self.object_indices[env_ids],
                    self.goal_object_indices[env_ids],
                ]
            ).to(torch.int32)
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices),
        )

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices),
        )

        all_indices = torch.unique(torch.cat([all_hand_indices, object_indices]).to(torch.int32))

        # self.init_step = random.randint(0, self.max_episode_length - 211)
        self.init_step = 0
        self.progress_buf[env_ids] = self.init_step
        # self.end_step_buf[env_ids] = self.init_step + 1
        self.end_step_buf[env_ids] = self.init_step + self.arctic_traj_len - 11
        self.r_pos_global_init[env_ids] = self.trans_r[env_ids, self.init_step]
        self.r_rot_global_init[env_ids] = self.rot_r_quat[env_ids, self.init_step]
        self.l_pos_global_init[env_ids] = self.trans_l[env_ids, self.init_step]
        self.l_rot_global_init[env_ids] = self.rot_l_quat[env_ids, self.init_step]
        self.obj_pos_global_init[env_ids] = self.obj_params[env_ids, self.init_step, 4:7]
        self.obj_rot_global_init[env_ids] = self.obj_rot_quat[env_ids, self.init_step, 0:4]

        self.root_state_tensor[self.object_indices[env_ids], 0:3] = self.obj_pos_global_init[
            env_ids
        ]
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = self.obj_rot_global_init[
            env_ids
        ]
        self.root_state_tensor[self.object_indices[env_ids], 7:10] = 0
        self.root_state_tensor[self.object_indices[env_ids], 10:13] = 0

        if self.used_hand_type != "mano_free":
            self.root_state_tensor[self.hand_indices[env_ids], 0:3] = self.r_pos_global_init[env_ids]
            self.root_state_tensor[self.hand_indices[env_ids], 3:7] = self.r_rot_global_init[env_ids]

            self.root_state_tensor[self.another_hand_indices[env_ids], 0:3] = self.l_pos_global_init[
                env_ids
            ]
            self.root_state_tensor[self.another_hand_indices[env_ids], 3:7] = self.l_rot_global_init[
                env_ids
            ]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_indices),
            len(all_indices),
        )
        # self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.last_actions[env_ids] = torch.zeros_like(self.actions[env_ids])

        self.proprioception_close_loop[env_ids] = self.allegro_hand_dof_pos[env_ids, 0:22].clone()

        self.object_state_stack_frames[env_ids] = torch.zeros_like(
            self.object_state_stack_frames[env_ids]
        )
        self.ran[env_ids] = torch.zeros_like(self.ran[env_ids])

    def pre_physics_step(self, actions):
        # if self.progress_buf[0] > 20:
        #     self.tarject_predict(0, 19)
        self.actions = actions.clone().to(self.device)

        # traj = np.loadtxt("/home/jmji/DexterousHandEnvs/dexteroushandenvs/trajectory/35_joint_big_1_1000density/right_allegro_hand_dof_pos.txt")
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if self.enable_camera_sensors:
            env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            
            for i in self.seq_list_i:
                if i in env_ids and self.total_steps > 0:
                    self.video_out_list[i].release()
                    self.seq_list_i.remove(i)
                    if len(self.seq_list_i) == 0:
                        cv2.destroyAllWindows()
                        exit()
                    
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            for i in self.seq_list_i:
                camera_rgba_image = self.camera_rgb_visulization(self.camera_tensors, env_id=i, is_depth_image=False)
                self.video_out_list[i].write(camera_rgba_image)   

            cv2.imshow("DEBUG_RGB_VIS", camera_rgba_image)
            cv2.waitKey(1)
                
            self.gym.end_access_image_tensors(self.sim)
        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        ############################################################
        # self.apply_forces[:, 0, :] = actions[:, 0:3] * self.dt * 100
        # self.apply_forces[:, 0 + 19, :] = actions[:, 24:27] * self.dt * 100
        # self.apply_torque[:, 0, :] = actions[:, 3:6] * self.dt * 0.01
        # self.apply_torque[:, 0 + 19, :] = actions[:, 27:30] * self.dt * 0.01

        # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces), gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)
        #############################################################

        self.r_pos_global = self.trans_r[self.seq_idx_tensor, self.progress_buf].clone().squeeze(0)
        self.r_rot_global = (
            self.rot_r_quat[self.seq_idx_tensor, self.progress_buf].clone().squeeze(0)
        )
        self.l_pos_global = self.trans_l[self.seq_idx_tensor, self.progress_buf].clone().squeeze(0)
        self.l_rot_global = (
            self.rot_l_quat[self.seq_idx_tensor, self.progress_buf].clone().squeeze(0)
        )

        self.left_fingertip_global = self.left_fingertip[self.seq_idx_tensor, self.progress_buf].clone().squeeze(0)
        self.right_fingertip_global = (
            self.right_fingertip[self.seq_idx_tensor, self.progress_buf].clone().squeeze(0)
        )
        self.left_middle_finger_global = self.left_middle_finger[self.seq_idx_tensor, self.progress_buf].clone().squeeze(0)
        self.right_middle_finger_global = (
            self.right_middle_finger[self.seq_idx_tensor, self.progress_buf].clone().squeeze(0)
        )
                
        self.obj_pos_global = (
            self.obj_params[self.seq_idx_tensor, self.progress_buf, 4:7].clone().squeeze(0)
        )
        self.obj_rot_global = (
            self.obj_rot_quat[self.seq_idx_tensor, self.progress_buf, 0:4].clone().squeeze(0)
        )
        self.obj_joint = (
            self.obj_params[self.seq_idx_tensor, self.progress_buf, 0].clone().squeeze(0)
        )

        # # noise
        # noise = torch_rand_float(-1.0, 1.0, (self.num_envs, 63), device=self.device)

        # for i, ran in enumerate([1, 2,3, 5, 3, 1, 4, 5]):
        #     self.ran[i, 0:3] += torch.tanh(noise[i, 0:3]) * 0.001 * ran
        #     self.ran[i, 3:6] += torch.tanh(noise[i, 3:6]) * 0.001 * ran
        #     self.ran[i, 6:10] += torch.tanh(noise[i, 6:10]) * 0.005 * ran
        #     self.ran[i, 10:14] += torch.tanh(noise[i, 10:14]) * 0.005 * ran

        #     r_pos_global[i, 0:3] += self.ran[i, 0:3]
        #     l_pos_global[i, 0:3] += self.ran[i, 3:6]
        #     r_rot_global[i, 0:4] += self.ran[i, 6:10]
        #     l_rot_global[i, 0:4] += self.ran[i, 10:14]

        ########## High Level Planner #############
        if self.test_high_level_planner:
            # bc
            if self.is_multi_object:
                if self.high_level_planner == "bc":
                    obs = {"obj_joint": self.obj_params[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 0:1], "obj_trans": self.obj_params[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 4:7], "obj_quat": self.obj_rot_quat[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 0:4], "obj_bps": torch.tensor([0], dtype=torch.float, device=self.device).repeat((self.traj_len, 1)).reshape(-1)}
                    act = self.policy(ob=obs)
                if self.high_level_planner == "bc_trans":
                    act = self.policy(ob=self.obs)
                    obs = {"obj_joint": self.obj_params[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 0:1].reshape(-1), "obj_trans": self.obj_params[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 4:7].reshape(-1), "obj_quat": self.obj_rot_quat[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 0:4].reshape(-1), "obj_bps": torch.tensor([0], dtype=torch.float, device=self.device).repeat((self.traj_len, 1)).reshape(-1)}

                    self.update_obs(obs, action=act, reset=False)
                    for k in obs:
                        self.obs_history[k].append(obs[k][None])
                    self.obs = self._get_stacked_obs_from_history()

            else:
                if self.high_level_planner == "bc":
                    obs = {"obj_joint": self.obj_params[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 0:1], "obj_trans": self.obj_params[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 4:7], "obj_quat": self.obj_rot_quat[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 0:4]}
                    act = self.policy(ob=obs)
                # bc-trans
                if self.high_level_planner == "bc_trans":
                    act = self.policy(ob=self.obs)
                    
                    obs = {"obj_joint": self.obj_params[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 0:1].reshape(-1), "obj_trans": self.obj_params[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 4:7].reshape(-1), "obj_quat": self.obj_rot_quat[0, self.progress_buf[0]+1:self.progress_buf[0]+self.traj_len+1, 0:4].reshape(-1)}

                    self.update_obs(obs, action=act, reset=False)
                    for k in obs:
                        self.obs_history[k].append(obs[k][None])
                    self.obs = self._get_stacked_obs_from_history()
                
            rot_r_quat = to_torch(act[0:self.traj_len*4], dtype=torch.float, device=self.device)
            trans_r = to_torch(act[self.traj_len*4:self.traj_len*(4+3)], dtype=torch.float, device=self.device)
            rot_l_quat = to_torch(act[self.traj_len*(4+3):self.traj_len*(4+3+4)], dtype=torch.float, device=self.device)
            trans_l = to_torch(act[self.traj_len*(4+3+4):self.traj_len*(4+3+4+3)], dtype=torch.float, device=self.device)

            self.l_pos_global = trans_l[0:3]
            self.l_rot_global = rot_l_quat[0:4]
            self.r_pos_global = trans_r[0:3]
            self.r_rot_global = rot_r_quat[0:4]
            
            if self.if_calc_evaluation_metric:
                self.position_error_tensor[:, self.progress_buf] = (torch.norm(self.trans_l[
                    self.seq_idx_tensor, self.progress_buf
                ] - trans_l[0:3], p=2, dim=-1) + torch.norm(self.trans_r[
                    self.seq_idx_tensor, self.progress_buf
                ] - trans_r[0:3], p=2, dim=-1)) / 2
                
                self.rotation_error_tensor[:, self.progress_buf] = (2.0 * torch.asin(
                    torch.clamp(torch.norm(quat_mul(
                    rot_r_quat[0:4].unsqueeze(0), quat_conjugate(self.rot_r_quat[self.seq_idx_tensor, self.progress_buf].squeeze(0))
                )[:, 0:3], p=2, dim=-1), max=1.0)
                ) + 2.0 * torch.asin(
                    torch.clamp(torch.norm(quat_mul(
                    rot_l_quat[0:4].unsqueeze(0), quat_conjugate(self.rot_l_quat[self.seq_idx_tensor, self.progress_buf].squeeze(0))
                )[:, 0:3], p=2, dim=-1), max=1.0)
                )) / 2
                
                print("position_error: ", self.position_error_tensor.mean())
                print("rotation_error: ", self.rotation_error_tensor.mean())
                
            # filter
            # if self.total_steps > 0:
            #     self.r_pos_global = torch.where(torch.norm(self.r_pos_global - self.last_r_pos_global, p=2, dim=-1) > 0.1, self.last_r_pos_global, self.r_pos_global)
                
            #     self.l_pos_global = torch.where(torch.norm(self.l_pos_global - self.last_l_pos_global, p=2, dim=-1) > 0.1, self.last_l_pos_global, self.l_pos_global)
                
            #     if calculate_frobenius_norm_of_rotation_difference(self.r_rot_global, self.last_r_rot_global, device=self.device) > 0.5:
            #         self.r_rot_global = self.last_r_rot_global.clone()
                
            #     if calculate_frobenius_norm_of_rotation_difference(self.l_rot_global, self.last_l_rot_global, device=self.device) > 0.5:
            #         self.l_rot_global = self.last_l_rot_global.clone()
                
            self.last_r_pos_global = self.r_pos_global.clone()
            self.last_r_rot_global = self.r_rot_global.clone()
            self.last_l_pos_global = self.l_pos_global.clone()
            self.last_l_rot_global = self.l_rot_global.clone()
        
        # filter
        # if self.total_steps > 0:
        #     self.obj_pos_global = torch.where(torch.norm(self.obj_pos_global - self.last_obj_pos_global, p=2, dim=-1) > 0.02, self.last_obj_pos_global, self.obj_pos_global)

        #     print(calculate_frobenius_norm_of_rotation_difference(self.obj_rot_global[0], self.last_obj_rot_global[0], device=self.device))

        #     if calculate_frobenius_norm_of_rotation_difference(self.obj_rot_global[0], self.last_obj_rot_global[0], device=self.device) > 0.5:
        #         self.obj_rot_global = self.last_obj_rot_global.clone()
                
        # self.last_obj_pos_global = self.obj_pos_global.clone()
        # self.last_obj_rot_global = self.obj_rot_global.clone()
        
        print(self.total_steps)

        if self.used_hand_type != "mano_free":
            self.root_state_tensor[self.object_indices, 0:3] = self.obj_pos_global
            self.root_state_tensor[self.object_indices, 3:7] = self.obj_rot_global

            self.root_state_tensor[self.hand_indices, 0:3] = self.r_pos_global
            self.root_state_tensor[self.hand_indices, 3:7] = self.r_rot_global

            self.root_state_tensor[self.another_hand_indices, 0:3] = self.l_pos_global
            self.root_state_tensor[self.another_hand_indices, 3:7] = self.l_rot_global

            object_indices = torch.unique(
                torch.cat([self.object_indices, self.goal_object_indices]).to(torch.int32)
            )

            self.object_dof_pos[:, 0] = self.obj_joint[:]
            
            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.root_state_tensor)
            )
            
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        
        if "mano" in self.used_hand_type:
            if self.use_joint_space_retargeting:
                left_hand_joint, right_hand_joint = self.ik_solver.bimanual_position_optimizer(torch.concat((self.left_fingertip_global[0, :].view(5, 3) - self.l_pos_global.view(3), self.left_middle_finger_global[0, :].view(5, 3) - self.l_pos_global.view(3)), dim=0), torch.concat((self.right_fingertip_global[0, :].view(5, 3) - self.r_pos_global.view(3), self.right_middle_finger_global[0, :].view(5, 3) - self.r_pos_global.view(3)), dim=0))
                left_hand_joint = to_torch(list(left_hand_joint), device=self.device)
                right_hand_joint = to_torch(list(right_hand_joint), device=self.device)
                
                self.dof_state.view(self.num_envs, -1, 2)[
                    :, self.num_allegro_hand_dofs: self.num_allegro_hand_dofs * 2, 0
                ] = left_hand_joint
                
                self.dof_state.view(self.num_envs, -1, 2)[
                    :, 0: self.num_allegro_hand_dofs, 0
                ] = right_hand_joint
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
                
            else:
                left_hand_joint, right_hand_joint = self.ik_solver.solve_ik(self.left_fingertip_global[0, :].view(5, 3), self.l_pos_global.view(3), self.l_rot_global.view(4), self.right_fingertip_global[0, :].view(5, 3), self.r_pos_global.view(3), self.r_rot_global.view(4))
                left_hand_joint = to_torch(list(left_hand_joint), device=self.device)
                right_hand_joint = to_torch(list(right_hand_joint), device=self.device)
                if self.used_hand_type == "mano_free":
                    self.cur_targets[:, 0:3] = self.r_pos_global
                    self.cur_targets[:, 3:6] = xyzw_quaternion_to_euler_xyz(self.r_rot_global)
                    self.cur_targets[:, 6:26] = right_hand_joint
                    
                    self.cur_targets[:, 0+self.num_allegro_hand_dofs:3+self.num_allegro_hand_dofs] = self.l_pos_global
                    self.cur_targets[:, 3+self.num_allegro_hand_dofs:6+self.num_allegro_hand_dofs] = xyzw_quaternion_to_euler_xyz(self.l_rot_global)
                    self.cur_targets[:, 6+self.num_allegro_hand_dofs:26+self.num_allegro_hand_dofs] = left_hand_joint
                    
                    self.cur_targets[:, self.num_allegro_hand_dofs*2:self.num_allegro_hand_dofs*2+1] = self.object_dof_pos
                    self.prev_targets[:] = self.cur_targets[:].clone()
                    
                    self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
                
                self.dof_state.view(self.num_envs, -1, 2)[
                    :, self.num_allegro_hand_dofs: self.num_allegro_hand_dofs * 2, 0
                ] = left_hand_joint
                    
                self.dof_state.view(self.num_envs, -1, 2)[
                    :, 0: self.num_allegro_hand_dofs, 0
                ] = right_hand_joint
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))


    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        q = quat_from_angle_axis(torch.tensor(0.0, device=self.device).view(1, 1).repeat(
            (self.num_envs, 1)
        ), self.x_unit_tensor)[0]
        # self.add_debug_lines(self.envs[0], self.left_fingertip_global[0, 0:3], q, line_width=2, line_length=0.1)
        # self.add_debug_lines(self.envs[0], self.left_fingertip_global[0, 3:6], q, line_width=2, line_length=0.1)
        # self.add_debug_lines(self.envs[0], self.left_fingertip_global[0, 6:9], q, line_width=2, line_length=0.1)
        # self.add_debug_lines(self.envs[0], self.left_fingertip_global[0, 9:12], q, line_width=2, line_length=0.1)
        # self.add_debug_lines(self.envs[0], self.left_fingertip_global[0, 12:15], q, line_width=2, line_length=0.1)
        # self.add_debug_lines(self.envs[0], self.right_fingertip_global[0, 0:3], q, line_width=2, line_length=0.1)
        # self.add_debug_lines(self.envs[0], self.right_fingertip_global[0, 3:6], q, line_width=2, line_length=0.1)
        # self.add_debug_lines(self.envs[0], self.right_fingertip_global[0, 6:9], q, line_width=2, line_length=0.1)
        # self.add_debug_lines(self.envs[0], self.right_fingertip_global[0, 9:12], q, line_width=2, line_length=0.1)
        # self.add_debug_lines(self.envs[0], self.right_fingertip_global[0, 12:15], q, line_width=2, line_length=0.1)

        # self.add_debug_lines(self.envs[0], self.palm_joint_pos[0], self.palm_joint_rot[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.ff_joint_pos[0], self.ff_joint_rot[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.mf_joint_pos[0], self.mf_joint_rot[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.rf_joint_pos[0], self.rf_joint_rot[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.lf_joint_pos[0], self.lf_joint_rot[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.th_joint_pos[0], self.th_joint_rot[0], line_width=2)

        # self.add_debug_lines(self.envs[0], self.object_pos[0], self.object_rot[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.allegro_left_hand_pos[0], self.allegro_left_hand_rot[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.allegro_right_hand_pos[0], self.allegro_right_hand_rot[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.r_pos_global_init[0], self.r_rot_global_init[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.l_pos_global_init[0], self.l_rot_global_init[0], line_width=2)

        # self.add_debug_lines(self.envs[0], self.root_state_tensor[self.hand_indices, 0:3][0], self.root_state_tensor[self.hand_indices, 3:7][0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.root_state_tensor[self.another_hand_indices, 0:3][0], self.root_state_tensor[self.another_hand_indices, 3:7][0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.object_base_pos[0], self.object_base_rot[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.r_pos_global[0], self.r_rot_global[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.l_pos_global[0], self.l_rot_global[0], line_width=2)
        # self.add_debug_lines(
        #     self.envs[0], self.allegro_left_hand_pos[0], self.allegro_left_hand_rot[0], line_width=2
        # )
        # self.add_debug_lines(
        #     self.envs[0],
        #     self.allegro_right_hand_pos[0],
        #     self.allegro_right_hand_rot[0],
        #     line_width=2,
        # )
        # self.add_debug_lines(self.envs[0], self.object_pos[0], self.object_rot[0], line_width=2)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(
                    self.envs[i],
                    self.allegro_hand_another_thmub_pos[i],
                    self.allegro_hand_another_thmub_rot[i],
                    line_width=2,
                )
                # self.add_debug_lines(self.envs[i], self.allegro_left_hand_pos[i], self.allegro_left_hand_rot[i])

    def add_debug_lines(self, env, pos, rot, line_width=1, line_length=0.2):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * line_length)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * line_length)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * line_length)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(
            self.viewer,
            env,
            line_width,
            [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]],
            [0.85, 0.1, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            env,
            line_width,
            [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]],
            [0.1, 0.85, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            env,
            line_width,
            [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]],
            [0.1, 0.1, 0.85],
        )

    #####################################################################
    ###=========================jit functions=========================###
    #####################################################################

    def camera_visulization(self, is_depth_image=False):
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_DEPTH
            )
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -1, 1)
            torch_depth_tensor = scale(
                torch_depth_tensor,
                to_torch([0], dtype=torch.float, device=self.device),
                to_torch([256], dtype=torch.float, device=self.device),
            )
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)

        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR
            )
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)

        return camera_image

    def rand_row(self, tensor, dim_needed):
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)), :]

    def sample_points(self, points, sample_num=1000, sample_mathed='furthest'):
        eff_points = points[points[:, 2] > 0.04]
        if eff_points.shape[0] < sample_num:
            eff_points = points
        if sample_mathed == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_mathed == 'furthest':
            sampled_points_id = pointnet2_utils.furthest_point_sample(
                eff_points.reshape(1, *eff_points.shape), sample_num
            )
            sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points

    def camera_rgb_visulization(self, camera_tensors, env_id=0, is_depth_image=False):
        torch_rgba_tensor = camera_tensors[env_id].clone()
        camera_image = torch_rgba_tensor.cpu().numpy()
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
        
        return camera_image

@torch.jit.script
def depth_image_to_point_cloud_GPU(
    camera_tensor,
    camera_view_matrix_inv,
    camera_proj_matrix,
    u,
    v,
    width: float,
    height: float,
    depth_bar: float,
    device: torch.device,
):
    # time1 = time.time()
    depth_buffer = camera_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = camera_view_matrix_inv

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection

    proj = camera_proj_matrix
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    centerU = width / 2
    centerV = height / 2

    Z = depth_buffer
    X = -(u - centerU) / width * Z * fu
    Y = (v - centerV) / height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position @ vinv

    points = position[:, 0:3]

    return points


@torch.jit.script
def compute_hand_reward(
    rew_buf,
    reset_buf,
    reset_goal_buf,
    progress_buf,
    successes,
    consecutive_successes,
    left_contact,
    right_contact,
    allegro_left_hand_pos,
    allegro_right_hand_pos,
    allegro_left_hand_rot,
    allegro_right_hand_rot,
    max_episode_length: float,
    object_pos,
    object_rot,
    target_pos,
    target_rot,
    allegro_left_hand_dof,
    allegro_right_hand_dof,
    object_dof,
    trans_r,
    trans_l,
    rot_r_quat,
    rot_l_quat,
    obj_params,
    obj_quat,
    dist_reward_sccalculate_frobenius_norm_of_rotation_differenceale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions,
    action_penalty_scale: float,
    a_hand_palm_pos,
    last_actions,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    end_step_buf,
    seq_idx_tensor,
    max_consecutive_successes: int,
    av_factor: float,
    ignore_z_rot: bool,
):
    object_pos_dist = torch.norm(
        object_pos - obj_params[seq_idx_tensor, progress_buf, 4:7].squeeze(0), p=2, dim=-1
    )
    object_quat_diff = quat_mul(
        object_rot, quat_conjugate(obj_quat[seq_idx_tensor, progress_buf].squeeze(0))
    )
    object_rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(object_quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    )
    object_joint_dist = torch.abs(
        object_dof[:, 0] - obj_params[seq_idx_tensor, progress_buf, 0].squeeze(0)
    )

    object_pos_dist = torch.clamp(object_pos_dist - 0.05, 0, None)
    object_rot_dist = torch.clamp(object_rot_dist - 0.1, 0, None)
    object_joint_dist = torch.clamp(object_joint_dist - 0.1, 0, None)

    left_hand_pos_dist = torch.norm(
        allegro_left_hand_pos - trans_l[seq_idx_tensor, progress_buf].squeeze(0), p=2, dim=-1
    )
    left_hand_quat_diff = quat_mul(
        allegro_left_hand_rot, quat_conjugate(rot_l_quat[seq_idx_tensor, progress_buf].squeeze(0))
    )
    left_hand_rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(left_hand_quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    )

    left_hand_pos_dist = torch.clamp(left_hand_pos_dist - 0.2, 0, None)
    left_hand_rot_dist = torch.clamp(left_hand_rot_dist - 0.5, 0, None)

    right_hand_pos_dist = torch.norm(
        allegro_right_hand_pos - trans_r[seq_idx_tensor, progress_buf].squeeze(0), p=2, dim=-1
    )
    right_hand_quat_diff = quat_mul(
        allegro_right_hand_rot, quat_conjugate(rot_r_quat[seq_idx_tensor, progress_buf].squeeze(0))
    )
    right_hand_rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(right_hand_quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    )

    right_hand_pos_dist = torch.clamp(right_hand_pos_dist - 0.2, 0, None)
    right_hand_rot_dist = torch.clamp(right_hand_rot_dist - 0.5, 0, None)

    left_contact_reward = torch.sum(left_contact**2, dim=-1) / 10
    right_contact_reward = torch.sum(right_contact**2, dim=-1) / 10

    object_reward = torch.exp(-1 * object_rot_dist - 20 * object_pos_dist - 1 * object_joint_dist)
    left_hand_reward = torch.exp(-1 * left_hand_rot_dist - 20 * left_hand_pos_dist)
    right_hand_reward = torch.exp(-1 * right_hand_rot_dist - 20 * right_hand_pos_dist)
    # left_hand_reward = torch.ones_like(object_reward)
    # right_hand_reward = torch.ones_like(object_reward)

    jittering_penalty = 0.003 * torch.sum(actions**2, dim=-1)

    # reward = object_reward * right_hand_reward * left_hand_reward - jittering_penalty
    reward = (
        4 * object_reward + 0.05 * right_hand_reward + 0.05 * left_hand_reward - jittering_penalty
    )

    print("object_reward: ", object_reward[0].item())
    print("right_hand_reward: ", right_hand_reward[0].item())
    print("left_hand_reward: ", left_hand_reward[0].item())

    print("object_rot_dist: ", object_rot_dist[0].item())
    print("right_hand_rot_dist: ", right_hand_rot_dist[0].item())
    print("left_hand_rot_dist: ", left_hand_rot_dist[0].item())

    print("object_pos_dist: ", object_pos_dist[0].item())
    print("right_hand_pos_dist: ", right_hand_pos_dist[0].item())
    print("left_hand_pos_dist: ", left_hand_pos_dist[0].item())

    print("object_joint_dist: ", object_joint_dist[0].item())
    print("jittering_penalty: ", jittering_penalty[0].item())

    # Check env termination conditions, including maximum success number
    resets = torch.where(object_pos[:, 2] <= -10.15, torch.ones_like(reset_buf), reset_buf)

    # resets = torch.where(object_pos_dist >= 0.1, torch.ones_like(resets), resets)
    # resets = torch.where(object_rot_dist >= 1.0, torch.ones_like(resets), resets)
    # resets = torch.where(object_joint_dist >= 0.5, torch.ones_like(resets), resets)

    # print(object_joint_dist)

    # resets = torch.where(left_hand_pos_dist >= 0.2, torch.ones_like(resets), resets)
    # resets = torch.where(left_hand_rot_dist >= 0.5, torch.ones_like(resets), resets)

    # resets = torch.where(right_hand_pos_dist >= 0.2, torch.ones_like(resets), resets)
    # resets = torch.where(right_hand_rot_dist >= 0.5, torch.ones_like(resets), resets)

    # reward = torch.where(resets == 1, reward - 10, reward)

    # hard constraint finger motion
    goal_resets = torch.where(
        object_pos[:, 2] <= -10, torch.ones_like(reset_goal_buf), reset_goal_buf
    )
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    resets = torch.where(progress_buf >= end_step_buf, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(
            progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward
        )

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(
        quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
        quat_from_angle_axis(rand0 * np.pi, z_unit_tensor),
    )
    return rot


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(j_eef, device, dpose, num_envs):
    # Set controller parameters
    # IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u

def interpolate_tensor(input_tensor, interpolate_time):
    """
    对输入的数据列表进行插值，返回插值后的新数据列表
    参数：
    input_tensor: 输入的张量，维度为(batch, 4)
    new_batch_size: 插值后的新批次大小
    返回值：
    插值后的新数据列表
    """
    batch_size = input_tensor.size(0)
    
    # 原始形状
    original_shape = input_tensor.size()
    # 新形状
    new_batch_size = (original_shape[0] - 1) * interpolate_time + original_shape[0]
    new_shape = (new_batch_size, original_shape[1])
    # 新数据列表
    # 对每个向量进行插值
    interpolated_data = np.zeros((new_shape[0], new_shape[1]))
    for i in range(new_shape[1]):
        # 对每个维度进行线性插值
        x = np.linspace(0, batch_size - 1, batch_size)  # 使用原始批次大小作为 x 数组的长度
        y = input_tensor[:, i].cpu().numpy()
        f = interp1d(x, y, kind='linear')
        new_x = np.linspace(0, batch_size - 1, new_batch_size)
        interpolated_data[:, i] = f(new_x)
    
    # 构建输出张量
    output_tensor = torch.tensor(interpolated_data, dtype=input_tensor.dtype, device=input_tensor.device)

    return output_tensor


def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    
    Args:
        q (torch.Tensor): 四元数，形状为 (4, )
        
    Returns:
        torch.Tensor: 旋转矩阵，形状为 (3, 3)
    """
    # 规范化四元数
    q = q / torch.norm(q)
    
    # 四元数的元素
    x, y, z, w = q
    
    # 计算旋转矩阵的元素
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    
    yy = y * y
    yz = y * z
    yw = y * w
    
    zz = z * z
    zw = z * w
    
    # 构建旋转矩阵
    R = torch.zeros(3, 3, device=q.device)
    R[0, 0] = 1 - 2 * (yy + zz)
    R[0, 1] = 2 * (xy - zw)
    R[0, 2] = 2 * (xz + yw)
    
    R[1, 0] = 2 * (xy + zw)
    R[1, 1] = 1 - 2 * (xx + zz)
    R[1, 2] = 2 * (yz - xw)
    
    R[2, 0] = 2 * (xz - yw)
    R[2, 1] = 2 * (yz + xw)
    R[2, 2] = 1 - 2 * (xx + yy)
    
    return R

def calculate_frobenius_norm_of_rotation_difference(q_pred, q_gt, device=torch.device('cuda:0')):
    """
    计算两个四元数表示的旋转之间的 Frobenius 范数
    
    Args:
        q_pred (torch.Tensor): 预测的四元数，形状为 (4, )
        q_gt (torch.Tensor): 真实的四元数，形状为 (4, )
        device (torch.device): 计算所使用的设备，默认为 CPU
        
    Returns:
        float: 旋转差的 Frobenius 范数
    """
    # 将四元数转换为旋转矩阵
    R_pred = quaternion_to_rotation_matrix(q_pred.to(device))
    R_gt = quaternion_to_rotation_matrix(q_gt.to(device))

    # 计算旋转矩阵的差
    diff = R_pred - R_gt

    # 计算差矩阵的 Frobenius 范数
    frobenius_norm = torch.norm(diff, p='fro')

    return frobenius_norm.item()

@torch.jit.script
def xyzw_quaternion_to_euler_xyz(quaternion):
    """
    Convert quaternion to euler angles in XYZ order.

    Parameters:
    quaternion (torch.Tensor): Tensor of quaternions of shape (batch_size, 4) with the order (x, y, z, w).

    Returns:
    torch.Tensor: Tensor of euler angles in XYZ order of shape (batch_size, 3) representing roll, pitch, and yaw.
    """
    x, y, z, w = quaternion.unbind(dim=-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


@torch.jit.script
def wxyz_quaternion_to_euler_xyz(quaternion):
    """
    Convert quaternion to euler angles in XYZ order.

    Parameters:
    quaternion (torch.Tensor): Tensor of quaternions of shape (batch_size, 4) with the order (x, y, z, w).

    Returns:
    torch.Tensor: Tensor of euler angles in XYZ order of shape (batch_size, 3) representing roll, pitch, and yaw.
    """
    w, x, y, z = quaternion.unbind(dim=-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)
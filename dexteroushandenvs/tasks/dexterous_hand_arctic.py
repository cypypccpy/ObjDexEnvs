# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from matplotlib.pyplot import axis
import numpy as np
import os
import random
import torch

from utils.torch_jit_utils import *

from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class DexterousHandArctic(BaseTask):
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

        self.dexterous_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.arctic_raw_data_path = self.cfg["env"]["arctic_raw_data_path"]
        self.arctic_processed_path = self.cfg["env"]["arctic_processed_path"]

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time / (control_freq_inv * self.sim_params.dt))
            )
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

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

        self.used_training_objects = self.cfg["env"]["used_training_objects"]
        self.used_hand_type = self.cfg["env"]["used_hand_type"]
        self.traj_index = self.cfg["env"]["traj_index"]
        
        if self.used_training_objects[0] == "all":
            self.used_training_objects = ["box", "capsulemachine", "espressomachine", "ketchup", "laptop", "microwave", "mixer", "notebook", "phone", "scissors", "waffleiron"]
        
        self.obs_type = self.cfg["env"]["observationType"]

        print("Obs type:", self.obs_type)

        self.num_point_cloud_feature_dim = 384
        self.one_frame_num_obs = 178
        self.num_obs_dict = {
            "full_state": 571,
        }

        self.contact_sensor_names = ["wrist_2_link", "wrist_1_link", "shoulder_link", "upper_arm_link", "forearm_link"]

        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 571

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 30

        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 60

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.enable_camera_sensors = self.cfg["env"]["enableCameraSensors"]
        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)
        self.num_envs = cfg["env"]["numEnvs"]
        
        # ablation study
        self.use_fingertip_reward = self.cfg["env"]["use_fingertip_reward"]
        self.use_hierarchy = self.cfg["env"]["use_hierarchy"]
        self.use_p_c_impro_loop = self.cfg["env"]["use_p_c_impro_loop"]

        super().__init__(cfg=self.cfg, enable_camera_sensors=self.enable_camera_sensors)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.5, -0.0, 1.2)
            cam_target = gymapi.Vec3(-0.5, -0.0, 0.2)

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
        
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dexterous_hand_dofs*2 + 2)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.another_dexterous_hand_default_dof_pos = torch.zeros(
            self.num_dexterous_hand_dofs, dtype=torch.float, device=self.device
        )

        self.another_dexterous_hand_default_dof_pos[:6] = to_torch(
            [3.4991441036750577, -1.310780687961321, -2.128748927522598, -2.84180679300243, -1.2157104341775433, 3.1342631916289605-3.1415],
            dtype=torch.float,
            device=self.device,
        )

        self.dexterous_hand_default_dof_pos = torch.zeros(
            self.num_dexterous_hand_dofs, dtype=torch.float, device=self.device
        )

        self.dexterous_hand_default_dof_pos[:6] = to_torch(
            [-0.4235312584306925, -1.8417856713793022, 2.1118022259904565, -0.26705746630618066, 1.1434836562123438, -3.150733285519455],
            dtype=torch.float,
            device=self.device,
        )

        self.object_default_dof_pos = to_torch(
            [self.obj_params[0, 0, 0]], dtype=torch.float, device=self.device
        )
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dexterous_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_dexterous_hand_dofs
        ]
        self.dexterous_hand_dof_pos = self.dexterous_hand_dof_state[..., 0]
        self.dexterous_hand_dof_vel = self.dexterous_hand_dof_state[..., 1]

        self.dexterous_hand_another_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, self.num_dexterous_hand_dofs : self.num_dexterous_hand_dofs * 2
        ]
        self.dexterous_hand_another_dof_pos = self.dexterous_hand_another_dof_state[..., 0]
        self.dexterous_hand_another_dof_vel = self.dexterous_hand_another_dof_state[..., 1]

        self.env_dof_state = self.dof_state.view(self.num_envs, -1, 2)

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, self.num_dexterous_hand_dofs * 2 : self.num_dexterous_hand_dofs * 2 + 1
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

        self.object_seq_len = 20
        self.object_state_stack_frames = torch.zeros(
            (self.num_envs, self.object_seq_len * 3), dtype=torch.float, device=self.device
        )
        
        if self.used_hand_type == "shadow":
            self.another_hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.another_hand_indices[0], "wrist", gymapi.DOMAIN_ENV
            )
            self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.hand_indices[0], "wrist", gymapi.DOMAIN_ENV
            )
        elif self.used_hand_type == "allegro":
            self.another_hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.another_hand_indices[0], "link_0.0", gymapi.DOMAIN_ENV
            )
            self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.hand_indices[0], "link_0.0", gymapi.DOMAIN_ENV
            )
        elif self.used_hand_type == "schunk":
            self.another_hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.another_hand_indices[0], "left_hand_k", gymapi.DOMAIN_ENV
            )
            self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.hand_indices[0], "right_hand_k", gymapi.DOMAIN_ENV
            )
        elif self.used_hand_type == "ability":
            self.another_hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.another_hand_indices[0], "index_L1", gymapi.DOMAIN_ENV
            )
            self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.hand_indices[0], "index_L1", gymapi.DOMAIN_ENV
            )
        print("hand_base_rigid_body_index: ", self.hand_base_rigid_body_index)
        print("another_hand_base_rigid_body_index: ", self.another_hand_base_rigid_body_index)

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
        self.obj_rot_global_init = self.obj_rot_quat[:, 0, 0:4].clone()
        self.obj_joint_init = self.obj_params[:, 0, 0:1].clone()

        self.max_episode_length = self.trans_r.shape[1]
        self.init_step_buf = torch.zeros_like(self.progress_buf)
        self.end_step_buf = torch.zeros_like(self.progress_buf)

        self.last_actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
        )

        self.dexterous_right_hand_pos = self.rigid_body_states[:, 6, 0:3]
        self.dexterous_right_hand_rot = self.rigid_body_states[:, 6, 3:7]
        self.dexterous_left_hand_pos = self.rigid_body_states[:, 6 + self.num_dexterous_hand_bodies, 0:3]
        self.dexterous_left_hand_rot = self.rigid_body_states[:, 6 + self.num_dexterous_hand_bodies, 3:7]

        self.train_teacher_policy = True
        self.apply_perturbation = False

        self.r2 = torch.tensor([[-0.999999,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  0.0,  0.999999],
         [ 0.0000e+00,  0.999999,  0.0]], device=self.device)
        self.sim_to_real_rotation_quaternion = torch.tensor([[0.0000, 0.0000, -0.7071, 0.7071]], device=self.device)
        self.sim_to_real_translation_matrix = torch.tensor([[0.0, -1.0, -0.0, -0.0],
                                            [1.0, 0.0, 0.0, 0.0],
                                            [0.0, -0.0, 1.0, 0],
                                            [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device)
        self.sim_to_real_object_quaternion = torch.tensor([[-0.0000, -0.0000, -0.3825,  0.9240]], device=self.device)
        
        self.left_hand_fingertip_pos_list = torch.zeros((5, self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.right_hand_fingertip_pos_list = torch.zeros((5, self.num_envs, 3), device=self.device, dtype=torch.float32)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

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
            object_asset_options.fix_base_link = False
            object_asset_options.flip_visual_attachments = False
            object_asset_options.collapse_fixed_joints = True
            object_asset_options.disable_gravity = False
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

        if self.used_hand_type == "shadow":
            dexterous_hand_asset_file = "urdf/shadow_hand_description/ur10e_shadowhand_right_digital_twin.urdf"
            dexterous_hand_another_asset_file = "urdf/shadow_hand_description/ur10e_shadowhand_left_digital_twin.urdf"
            self.hand_fingertip_index = [12, 17, 21, 25, 30]
            self.hand_fingertip_offset = 0.02
            self.hand_base_offset = 0.36

        elif self.used_hand_type == "allegro":
            dexterous_hand_asset_file = "urdf/shadow_hand_description/ur10e_allegrohand_right_digital_twin.urdf"
            dexterous_hand_another_asset_file = "urdf/shadow_hand_description/ur10e_allegrohand_left_digital_twin.urdf"
            self.hand_fingertip_index = [10, 14, 18, 22]
            self.hand_fingertip_offset = 0.02
            self.hand_base_offset = 0.2

        elif self.used_hand_type == "schunk":
            dexterous_hand_asset_file = "urdf/shadow_hand_description/ur10e_schunkhand_right_digital_twin.urdf"
            dexterous_hand_another_asset_file = "urdf/shadow_hand_description/ur10e_schunkhand_left_digital_twin.urdf"
            self.hand_fingertip_index = [9, 13, 17, 22, 26]
            self.hand_fingertip_offset = 0.0
            self.hand_base_offset = 0.18

        elif self.used_hand_type == "ability":
            dexterous_hand_asset_file = "urdf/shadow_hand_description/ur10e_abilityhand_right_digital_twin.urdf"
            dexterous_hand_another_asset_file = "urdf/shadow_hand_description/ur10e_abilityhand_left_digital_twin.urdf"
            self.hand_fingertip_index = [8, 10, 12, 12, 16]
            self.hand_fingertip_offset = 0.0
            self.hand_base_offset = 0.13

        else:
            raise Exception(
        "Unrecognized hand type!\Hand type should be one of: [shadow, dexterous, schunk, ability]"
    )
        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        dexterous_hand_asset = self.gym.load_asset(
            self.sim, asset_root, dexterous_hand_asset_file, asset_options
        )
        dexterous_hand_another_asset = self.gym.load_asset(
            self.sim, asset_root, dexterous_hand_another_asset_file, asset_options
        )

        self.num_dexterous_hand_bodies = self.gym.get_asset_rigid_body_count(dexterous_hand_asset)
        self.num_dexterous_hand_shapes = self.gym.get_asset_rigid_shape_count(dexterous_hand_asset)
        self.num_dexterous_hand_dofs = self.gym.get_asset_dof_count(dexterous_hand_asset)
        self.num_dexterous_hand_actuators = self.gym.get_asset_dof_count(dexterous_hand_asset)
        self.num_dexterous_hand_tendons = self.gym.get_asset_tendon_count(dexterous_hand_asset)

        print("self.num_dexterous_hand_bodies: ", self.num_dexterous_hand_bodies)
        print("self.num_dexterous_hand_shapes: ", self.num_dexterous_hand_shapes)
        print("self.num_dexterous_hand_dofs: ", self.num_dexterous_hand_dofs)
        print("self.num_dexterous_hand_actuators: ", self.num_dexterous_hand_actuators)
        print("self.num_dexterous_hand_tendons: ", self.num_dexterous_hand_tendons)

        self.actuated_dof_indices = [i for i in range(16)]

        # set dexterous_hand dof properties
        dexterous_hand_dof_props = self.gym.get_asset_dof_properties(dexterous_hand_asset)
        dexterous_hand_another_dof_props = self.gym.get_asset_dof_properties(
            dexterous_hand_another_asset
        )

        self.dexterous_hand_dof_lower_limits = []
        self.dexterous_hand_dof_upper_limits = []
        self.a_dexterous_hand_dof_lower_limits = []
        self.a_dexterous_hand_dof_upper_limits = []
        self.dexterous_hand_dof_default_pos = []
        self.dexterous_hand_dof_default_vel = []
        self.dexterous_hand_dof_stiffness = []
        self.dexterous_hand_dof_damping = []
        self.dexterous_hand_dof_effort = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_dexterous_hand_dofs):
            self.dexterous_hand_dof_lower_limits.append(dexterous_hand_dof_props['lower'][i])
            self.dexterous_hand_dof_upper_limits.append(dexterous_hand_dof_props['upper'][i])
            self.a_dexterous_hand_dof_lower_limits.append(dexterous_hand_another_dof_props['lower'][i])
            self.a_dexterous_hand_dof_upper_limits.append(dexterous_hand_another_dof_props['upper'][i])
            self.dexterous_hand_dof_default_pos.append(0.0)
            self.dexterous_hand_dof_default_vel.append(0.0)

            dexterous_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_NONE
            dexterous_hand_another_dof_props['driveMode'][i] = gymapi.DOF_MODE_NONE
            if i < 6:
                dexterous_hand_dof_props['stiffness'][i] = 1000
                dexterous_hand_dof_props['effort'][i] = 2000
                dexterous_hand_dof_props['damping'][i] = 100
                dexterous_hand_dof_props['velocity'][i] = 4
                dexterous_hand_another_dof_props['stiffness'][i] = 1000
                dexterous_hand_another_dof_props['effort'][i] = 2000
                dexterous_hand_another_dof_props['damping'][i] = 100
                dexterous_hand_another_dof_props['velocity'][i] = 4

            else:
                dexterous_hand_dof_props['velocity'][i] = 3.0
                dexterous_hand_dof_props['stiffness'][i] = 30
                dexterous_hand_dof_props['effort'][i] = 5
                dexterous_hand_dof_props['damping'][i] = 1
                dexterous_hand_another_dof_props['velocity'][i] = 3.0
                dexterous_hand_another_dof_props['stiffness'][i] = 30
                dexterous_hand_another_dof_props['effort'][i] = 5
                dexterous_hand_another_dof_props['damping'][i] = 1

            if self.used_hand_type == "shadow":
                if 8 > i > 6:
                    dexterous_hand_dof_props['velocity'][i] = 3.0
                    dexterous_hand_dof_props['stiffness'][i] = 150
                    dexterous_hand_dof_props['effort'][i] = 25
                    dexterous_hand_dof_props['damping'][i] = 10
                    dexterous_hand_another_dof_props['velocity'][i] = 3.0
                    dexterous_hand_another_dof_props['stiffness'][i] = 150
                    dexterous_hand_another_dof_props['effort'][i] = 25
                    dexterous_hand_another_dof_props['damping'][i] = 10

        self.actuated_dof_indices = to_torch(
            self.actuated_dof_indices, dtype=torch.long, device=self.device
        )
        self.dexterous_hand_dof_lower_limits = to_torch(
            self.dexterous_hand_dof_lower_limits, device=self.device
        )
        self.dexterous_hand_dof_upper_limits = to_torch(
            self.dexterous_hand_dof_upper_limits, device=self.device
        )
        self.a_dexterous_hand_dof_lower_limits = to_torch(
            self.a_dexterous_hand_dof_lower_limits, device=self.device
        )
        self.a_dexterous_hand_dof_upper_limits = to_torch(
            self.a_dexterous_hand_dof_upper_limits, device=self.device
        )
        self.dexterous_hand_dof_default_pos = to_torch(
            self.dexterous_hand_dof_default_pos, device=self.device
        )
        self.dexterous_hand_dof_default_vel = to_torch(
            self.dexterous_hand_dof_default_vel, device=self.device
        )

        self.object_name = self.used_training_objects
                
        if self.traj_index == "all":
            self.functional = ["use", "grab"]
            used_seq_list = ["01", "02", "04", "07", "08", "09"]
            used_sub_seq_list = ["01", "02", "03", "04"]
 
        else:
            self.functional = ["use"]
            if self.used_training_objects[0] == "capsulemachine" or self.used_training_objects[0] == "laptop":
                self.functional = ["grab"]
            
            if self.traj_index.split("_")[-1] == "grab":
                self.functional = ["grab"]
            
            used_seq_list = [self.traj_index.split("_")[0]] 
            used_sub_seq_list = [self.traj_index.split("_")[1]]         
            
        from high_level_planner.data_utils import DataLoader
        self.interpolate_time = 1

        self.dl = DataLoader(self.gym, self.sim, used_seq_list, self.functional, used_sub_seq_list, self.object_name, self.device, interpolate_time=self.interpolate_time)
        self.seq_list, self.texture_list, self.obj_name_seq = self.dl.load_arctic_data(self.arctic_processed_path, self.arctic_raw_data_path)

        print("seq_num: ", len(self.seq_list))
        self.seq_list_i = [i for i in range(len(self.seq_list))]

        self.traj_len = 1000
            
        self.rot_r = torch.zeros((self.num_envs, self.traj_len, 3), device=self.device, dtype=torch.float)
        self.trans_r = torch.zeros((self.num_envs, self.traj_len, 3), device=self.device, dtype=torch.float)
        self.rot_l = torch.zeros((self.num_envs, self.traj_len, 3), device=self.device, dtype=torch.float)
        self.trans_l = torch.zeros((self.num_envs, self.traj_len, 3), device=self.device, dtype=torch.float)
        self.obj_params = torch.zeros(
            (self.num_envs, self.traj_len, 7), device=self.device, dtype=torch.float
        )
        self.obj_rot_quat = torch.zeros(
            (self.num_envs, self.traj_len, 4), device=self.device, dtype=torch.float
        )
        self.rot_r_quat = torch.zeros(
            (self.num_envs, self.traj_len, 4), device=self.device, dtype=torch.float
        )
        self.rot_l_quat = torch.zeros(
            (self.num_envs, self.traj_len, 4), device=self.device, dtype=torch.float
        )
        self.left_fingertip = torch.zeros(
            (self.num_envs, self.traj_len, 15), device=self.device, dtype=torch.float
        )
        self.right_fingertip = torch.zeros(
            (self.num_envs, self.traj_len, 15), device=self.device, dtype=torch.float
        )
        self.left_middle_finger = torch.zeros(
            (self.num_envs, self.traj_len, 15), device=self.device, dtype=torch.float
        )
        self.right_middle_finger = torch.zeros(
            (self.num_envs, self.traj_len, 15), device=self.device, dtype=torch.float
        )
        
        for i in range(self.num_envs):
            seq_idx = i % len(self.seq_list)
            self.seq_idx_tensor = to_torch([range(self.num_envs)], dtype=int, device=self.device)
            self.rot_r[i] = self.seq_list[seq_idx]["rot_r"][:self.traj_len].clone()
            self.trans_r[i] = self.seq_list[seq_idx]["trans_r"][:self.traj_len].clone()
            self.rot_l[i] = self.seq_list[seq_idx]["rot_l"][:self.traj_len].clone()
            self.trans_l[i] = self.seq_list[seq_idx]["trans_l"][:self.traj_len].clone()
            self.obj_params[i] = self.seq_list[seq_idx]["obj_params"][:self.traj_len].clone()
            self.obj_rot_quat[i] = self.seq_list[seq_idx]["obj_rot_quat"][:self.traj_len].clone()
            self.rot_r_quat[i] = self.seq_list[seq_idx]["rot_r_quat"][:self.traj_len].clone()
            self.rot_l_quat[i] = self.seq_list[seq_idx]["rot_l_quat"][:self.traj_len].clone()
            self.left_fingertip[i] = self.seq_list[seq_idx]["left_fingertip"][:self.traj_len].clone()
            self.right_fingertip[i] = self.seq_list[seq_idx]["right_fingertip"][:self.traj_len].clone()
            self.left_middle_finger[i] = self.seq_list[seq_idx]["left_middle_finger"][:self.traj_len].clone()
            self.right_middle_finger[i] = self.seq_list[seq_idx]["right_middle_finger"][:self.traj_len].clone()
            
        if self.used_training_objects[0] == "espressomachine":
            self.object_joint_tolerate = 0.05
            self.object_joint_reset = 0.2
        else:
            self.object_joint_tolerate = 0.1
            self.object_joint_reset = 0.5
            
        dexterous_hand_start_pose = gymapi.Transform()
        dexterous_hand_start_pose.p = gymapi.Vec3(-0.5, 0.95, 0.7)
        dexterous_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.571)

        dexterous_another_hand_start_pose = gymapi.Transform()
        dexterous_another_hand_start_pose.p = gymapi.Vec3(0.5, 0.95, 0.7)
        dexterous_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.571)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0, -0.25, 0)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0.0)

        self.goal_displacement = gymapi.Vec3(-0.0, 0.0, 0.0)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z],
            device=self.device,
        )
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.0

        # create table asset
        table_dims = gymapi.Vec3(1.5, 2.2, 0.76)
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
        if self.used_training_objects[0] == "scissors":
            support_box_pose.p = gymapi.Vec3(0.0, 0.00, 0.5 * (2 * table_dims.z + support_box_dims.z))
        if self.used_training_objects[0] in ["espressomachine", "microwave"]:
            support_box_pose.p = gymapi.Vec3(0.1, -0.10, 0.5 * (2 * table_dims.z + support_box_dims.z))
        support_box_pose.r = gymapi.Quat().from_euler_zyx(-0, 0, 0.0)

        # compute aggregate size
        max_agg_bodies = self.num_dexterous_hand_bodies * 2 + 2 + 50
        max_agg_shapes = self.num_dexterous_hand_shapes * 2 + 2 + 50

        self.dexterous_hands = []
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

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            dexterous_hand_actor = self.gym.create_actor(
                env_ptr, dexterous_hand_asset, dexterous_hand_start_pose, "hand", i, 0, 0
            )
            dexterous_hand_another_actor = self.gym.create_actor(
                env_ptr,
                dexterous_hand_another_asset,
                dexterous_another_hand_start_pose,
                "another_hand",
                i,
                0,
                0,
            )

            self.hand_start_states.append(
                [
                    dexterous_hand_start_pose.p.x,
                    dexterous_hand_start_pose.p.y,
                    dexterous_hand_start_pose.p.z,
                    dexterous_hand_start_pose.r.x,
                    dexterous_hand_start_pose.r.y,
                    dexterous_hand_start_pose.r.z,
                    dexterous_hand_start_pose.r.w,
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
                    dexterous_another_hand_start_pose.p.x,
                    dexterous_another_hand_start_pose.p.y,
                    dexterous_another_hand_start_pose.p.z,
                    dexterous_another_hand_start_pose.r.x,
                    dexterous_another_hand_start_pose.r.y,
                    dexterous_another_hand_start_pose.r.z,
                    dexterous_another_hand_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )

            self.gym.set_actor_dof_properties(env_ptr, dexterous_hand_actor, dexterous_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, dexterous_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            self.gym.set_actor_dof_properties(
                env_ptr, dexterous_hand_another_actor, dexterous_hand_another_dof_props
            )
            another_hand_idx = self.gym.get_actor_index(
                env_ptr, dexterous_hand_another_actor, gymapi.DOMAIN_SIM
            )
            self.another_hand_indices.append(another_hand_idx)

            self.gym.enable_actor_dof_force_sensors(env_ptr, dexterous_hand_actor)
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexterous_hand_another_actor)
            
            # add object
            index = i % len(self.obj_name_seq)
            select_obj = self.obj_name_seq[index]

            object_handle = self.gym.create_actor(
                env_ptr,
                self.object_asset_dict[select_obj]['obj'],
                object_start_pose,
                "object",
                i,
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
                lego_body_prop.mass *= 1.0
                if self.used_training_objects[0] == "notebook":
                    lego_body_prop.mass *= 1.5
            self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, lego_body_props)

            object_dof_props = self.gym.get_actor_dof_properties(env_ptr, object_handle)
            for object_dof_prop in object_dof_props:
                object_dof_prop[4] = 100
                object_dof_prop[5] = 50
                object_dof_prop[6] = 5
                object_dof_prop[7] = 1
            self.gym.set_actor_dof_properties(env_ptr, object_handle, object_dof_props)

            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            for object_shape_prop in object_shape_props:
                object_shape_prop.restitution = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)
            
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

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.dexterous_hands.append(dexterous_hand_actor)

        another_sensor_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, dexterous_hand_another_actor, sensor_name)
            for sensor_name in self.contact_sensor_names
        ]

        sensor_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, dexterous_hand_actor, sensor_name)
            for sensor_name in self.contact_sensor_names
        ]

        object_sensor_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, object_handle, sensor_name)
            for sensor_name in ["bottom", "top"]
        ]

        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64, device=self.device)
        self.another_sensor_handle_indices = to_torch(another_sensor_handles, dtype=torch.int64, device=self.device)
        self.object_sensor_handles_indices = to_torch(object_sensor_handles, dtype=torch.int64, device=self.device)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
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
            self.object_contacts,
            self.left_contacts,
            self.right_contacts,
            self.dexterous_left_hand_pos,
            self.dexterous_right_hand_pos,
            self.dexterous_left_hand_rot,
            self.dexterous_right_hand_rot,
            self.max_episode_length,
            self.object_base_pos,
            self.object_base_rot,
            self.goal_pos,
            self.goal_rot,
            self.dexterous_left_hand_dof,
            self.dexterous_right_hand_dof,
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
            self.right_hand_energy_penalty,
            self.left_hand_energy_penalty,
            self.end_step_buf,
            self.seq_idx_tensor,
            self.max_consecutive_successes,
            self.av_factor,
            (False),
            self.object_joint_tolerate,
            self.object_joint_reset,
            self.use_fingertip_reward,
            self.use_hierarchy,
            self.left_fingertip_global,
            self.right_fingertip_global,
            self.left_hand_fingertip_pos_list,
            self.right_hand_fingertip_pos_list,
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
        self.gym.refresh_dof_force_tensor(self.sim)

        self.dexterous_right_hand_base_pos = self.root_state_tensor[self.hand_indices, 0:3]
        self.dexterous_right_hand_base_rot = self.root_state_tensor[self.hand_indices, 3:7]

        self.dexterous_left_hand_base_pos = self.root_state_tensor[self.another_hand_indices, 0:3]
        self.dexterous_left_hand_base_rot = self.root_state_tensor[self.another_hand_indices, 3:7]

        self.object_base_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_base_rot = self.root_state_tensor[self.object_indices, 3:7]

        self.dexterous_right_hand_pos = self.rigid_body_states[:, 6, 0:3]
        self.dexterous_right_hand_rot = self.rigid_body_states[:, 6, 3:7]
        self.dexterous_right_hand_linvel = self.rigid_body_states[:, 6, 7:10]
        self.dexterous_right_hand_angvel = self.rigid_body_states[:, 6, 10:13]

        self.dexterous_left_hand_pos = self.rigid_body_states[:, 6 + self.num_dexterous_hand_bodies, 0:3]
        self.dexterous_left_hand_rot = self.rigid_body_states[:, 6 + self.num_dexterous_hand_bodies, 3:7]
        self.dexterous_left_hand_linvel = self.rigid_body_states[:, 6 + self.num_dexterous_hand_bodies, 7:10]
        self.dexterous_left_hand_angvel = self.rigid_body_states[:, 6 + self.num_dexterous_hand_bodies, 10:13]

        self.a_hand_palm_pos = self.dexterous_left_hand_pos.clone()
        
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.object_bottom_pose = self.rigid_body_states[:, self.num_dexterous_hand_bodies*2 + 0, 0:7]
        self.object_bottom_pos = self.rigid_body_states[:, self.num_dexterous_hand_bodies*2 + 0, 0:3]
        self.object_bottom_rot = self.rigid_body_states[:, self.num_dexterous_hand_bodies*2 + 0, 3:7]
        
        self.object_bottom_rot = quaternion_multiply(self.object_bottom_rot, self.sim_to_real_object_quaternion)
        self.object_bottom_pos = self.object_bottom_pos + quat_apply(self.object_bottom_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.object_bottom_pos = self.object_bottom_pos + quat_apply(self.object_bottom_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.13)
        self.object_bottom_pos = self.object_bottom_pos + quat_apply(self.object_bottom_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * -0.03)
        
        self.object_top_pose = self.rigid_body_states[:, self.num_dexterous_hand_bodies*2 + 1, 0:7]
        self.object_top_pos = self.rigid_body_states[:, self.num_dexterous_hand_bodies*2 + 1, 0:3]
        self.object_top_rot = self.rigid_body_states[:, self.num_dexterous_hand_bodies*2 + 1, 3:7]

        self.object_top_rot = quaternion_multiply(self.object_top_rot, self.sim_to_real_object_quaternion)
        self.object_top_pos = self.object_top_pos + quat_apply(self.object_top_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.object_top_pos = self.object_top_pos + quat_apply(self.object_top_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.13)
        self.object_top_pos = self.object_top_pos + quat_apply(self.object_top_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.03)

        self.dexterous_right_hand_dof = self.dexterous_hand_dof_pos.clone()
        self.dexterous_left_hand_dof = self.dexterous_hand_another_dof_pos.clone()
        self.object_dof = self.object_dof_pos.clone()

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        
        # right hand finger
        if self.used_hand_type in ["shadow", "schunk", "ability"]:
            self.dexterous_right_hand_pos = self.dexterous_right_hand_pos + quat_apply(self.dexterous_right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * self.hand_base_offset).clone()
            a = quaternion_to_rotation_matrix(self.dexterous_right_hand_rot)
            c = a @ self.r2
            self.dexterous_right_hand_rot = rotation_matrix_to_quaternion(c)
            
            self.dexterous_left_hand_pos = self.dexterous_left_hand_pos + quat_apply(self.dexterous_left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * self.hand_base_offset).clone()
            a = quaternion_to_rotation_matrix(self.dexterous_left_hand_rot)
            c = a @ self.r2
            self.dexterous_left_hand_rot = rotation_matrix_to_quaternion(c)
        
            self.right_hand_ff_pos = self.rigid_body_states[:, self.hand_fingertip_index[0], 0:3]
            self.right_hand_ff_rot = self.rigid_body_states[:, self.hand_fingertip_index[0], 3:7]
            self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.right_hand_lf_pos = self.rigid_body_states[:, self.hand_fingertip_index[1], 0:3]
            self.right_hand_lf_rot = self.rigid_body_states[:, self.hand_fingertip_index[1], 3:7]
            self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.right_hand_mf_pos = self.rigid_body_states[:, self.hand_fingertip_index[2], 0:3]
            self.right_hand_mf_rot = self.rigid_body_states[:, self.hand_fingertip_index[2], 3:7]
            self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.right_hand_rf_pos = self.rigid_body_states[:, self.hand_fingertip_index[3], 0:3]
            self.right_hand_rf_rot = self.rigid_body_states[:, self.hand_fingertip_index[3], 3:7]
            self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.right_hand_th_pos = self.rigid_body_states[:, self.hand_fingertip_index[4], 0:3]
            self.right_hand_th_rot = self.rigid_body_states[:, self.hand_fingertip_index[4], 3:7]
            self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)

            self.right_hand_ff_state = self.rigid_body_states[:, self.hand_fingertip_index[0], 0:13]
            self.right_hand_lf_state = self.rigid_body_states[:, self.hand_fingertip_index[1], 0:13]
            self.right_hand_mf_state = self.rigid_body_states[:, self.hand_fingertip_index[2], 0:13]
            self.right_hand_rf_state = self.rigid_body_states[:, self.hand_fingertip_index[3], 0:13]
            self.right_hand_th_state = self.rigid_body_states[:, self.hand_fingertip_index[4], 0:13]

            self.left_hand_ff_pos = self.rigid_body_states[:, self.hand_fingertip_index[0] + self.num_dexterous_hand_bodies, 0:3]
            self.left_hand_ff_rot = self.rigid_body_states[:, self.hand_fingertip_index[0] + self.num_dexterous_hand_bodies, 3:7]
            self.left_hand_ff_pos = self.left_hand_ff_pos + quat_apply(self.left_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.left_hand_lf_pos = self.rigid_body_states[:, self.hand_fingertip_index[1] + self.num_dexterous_hand_bodies, 0:3]
            self.left_hand_lf_rot = self.rigid_body_states[:, self.hand_fingertip_index[1] + self.num_dexterous_hand_bodies, 3:7]
            self.left_hand_lf_pos = self.left_hand_lf_pos + quat_apply(self.left_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.left_hand_mf_pos = self.rigid_body_states[:, self.hand_fingertip_index[2] + self.num_dexterous_hand_bodies, 0:3]
            self.left_hand_mf_rot = self.rigid_body_states[:, self.hand_fingertip_index[2] + self.num_dexterous_hand_bodies, 3:7]
            self.left_hand_mf_pos = self.left_hand_mf_pos + quat_apply(self.left_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.left_hand_rf_pos = self.rigid_body_states[:, self.hand_fingertip_index[3] + self.num_dexterous_hand_bodies, 0:3]
            self.left_hand_rf_rot = self.rigid_body_states[:, self.hand_fingertip_index[3] + self.num_dexterous_hand_bodies, 3:7]
            self.left_hand_rf_pos = self.left_hand_rf_pos + quat_apply(self.left_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.left_hand_th_pos = self.rigid_body_states[:, self.hand_fingertip_index[4] + self.num_dexterous_hand_bodies, 0:3]
            self.left_hand_th_rot = self.rigid_body_states[:, self.hand_fingertip_index[4] + self.num_dexterous_hand_bodies, 3:7]
            self.left_hand_th_pos = self.left_hand_th_pos + quat_apply(self.left_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)

            self.left_hand_ff_state = self.rigid_body_states[:, self.hand_fingertip_index[0] + self.num_dexterous_hand_bodies, 0:13]
            self.left_hand_lf_state = self.rigid_body_states[:, self.hand_fingertip_index[1] + self.num_dexterous_hand_bodies, 0:13]
            self.left_hand_mf_state = self.rigid_body_states[:, self.hand_fingertip_index[2] + self.num_dexterous_hand_bodies, 0:13]
            self.left_hand_rf_state = self.rigid_body_states[:, self.hand_fingertip_index[3] + self.num_dexterous_hand_bodies, 0:13]
            self.left_hand_th_state = self.rigid_body_states[:, self.hand_fingertip_index[4] + self.num_dexterous_hand_bodies, 0:13]

        elif self.used_hand_type == "allegro":
            self.dexterous_right_hand_pos = self.dexterous_right_hand_pos + quat_apply(self.dexterous_right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * self.hand_base_offset).clone()
            a = quaternion_to_rotation_matrix(self.dexterous_right_hand_rot)
            c = a @ self.r2
            self.dexterous_right_hand_rot = rotation_matrix_to_quaternion(c)
            
            self.dexterous_left_hand_pos = self.dexterous_left_hand_pos + quat_apply(self.dexterous_left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * self.hand_base_offset).clone()
            a = quaternion_to_rotation_matrix(self.dexterous_left_hand_rot)
            c = a @ self.r2
            self.dexterous_left_hand_rot = rotation_matrix_to_quaternion(c)
        
            self.right_hand_ff_pos = self.rigid_body_states[:, self.hand_fingertip_index[0], 0:3]
            self.right_hand_ff_rot = self.rigid_body_states[:, self.hand_fingertip_index[0], 3:7]
            self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.right_hand_lf_pos = self.rigid_body_states[:, self.hand_fingertip_index[1], 0:3]
            self.right_hand_lf_rot = self.rigid_body_states[:, self.hand_fingertip_index[1], 3:7]
            self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.right_hand_rf_pos = self.rigid_body_states[:, self.hand_fingertip_index[2], 0:3]
            self.right_hand_rf_rot = self.rigid_body_states[:, self.hand_fingertip_index[2], 3:7]
            self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.right_hand_th_pos = self.rigid_body_states[:, self.hand_fingertip_index[3], 0:3]
            self.right_hand_th_rot = self.rigid_body_states[:, self.hand_fingertip_index[3], 3:7]
            self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)

            self.right_hand_ff_state = self.rigid_body_states[:, self.hand_fingertip_index[0], 0:13]
            self.right_hand_lf_state = self.rigid_body_states[:, self.hand_fingertip_index[1], 0:13]
            self.right_hand_rf_state = self.rigid_body_states[:, self.hand_fingertip_index[2], 0:13]
            self.right_hand_th_state = self.rigid_body_states[:, self.hand_fingertip_index[3], 0:13]

            self.left_hand_ff_pos = self.rigid_body_states[:, self.hand_fingertip_index[0] + self.num_dexterous_hand_bodies, 0:3]
            self.left_hand_ff_rot = self.rigid_body_states[:, self.hand_fingertip_index[0] + self.num_dexterous_hand_bodies, 3:7]
            self.left_hand_ff_pos = self.left_hand_ff_pos + quat_apply(self.left_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.left_hand_lf_pos = self.rigid_body_states[:, self.hand_fingertip_index[1] + self.num_dexterous_hand_bodies, 0:3]
            self.left_hand_lf_rot = self.rigid_body_states[:, self.hand_fingertip_index[1] + self.num_dexterous_hand_bodies, 3:7]
            self.left_hand_lf_pos = self.left_hand_lf_pos + quat_apply(self.left_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.left_hand_rf_pos = self.rigid_body_states[:, self.hand_fingertip_index[2] + self.num_dexterous_hand_bodies, 0:3]
            self.left_hand_rf_rot = self.rigid_body_states[:, self.hand_fingertip_index[2] + self.num_dexterous_hand_bodies, 3:7]
            self.left_hand_rf_pos = self.left_hand_rf_pos + quat_apply(self.left_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)
            self.left_hand_th_pos = self.rigid_body_states[:, self.hand_fingertip_index[3] + self.num_dexterous_hand_bodies, 0:3]
            self.left_hand_th_rot = self.rigid_body_states[:, self.hand_fingertip_index[3] + self.num_dexterous_hand_bodies, 3:7]
            self.left_hand_th_pos = self.left_hand_th_pos + quat_apply(self.left_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * self.hand_fingertip_offset)

            self.left_hand_ff_state = self.rigid_body_states[:, self.hand_fingertip_index[0] + self.num_dexterous_hand_bodies, 0:13]
            self.left_hand_lf_state = self.rigid_body_states[:, self.hand_fingertip_index[1] + self.num_dexterous_hand_bodies, 0:13]
            self.left_hand_rf_state = self.rigid_body_states[:, self.hand_fingertip_index[2] + self.num_dexterous_hand_bodies, 0:13]
            self.left_hand_th_state = self.rigid_body_states[:, self.hand_fingertip_index[3] + self.num_dexterous_hand_bodies, 0:13]

        self.right_hand_fingertip_pos_list[0] = self.right_hand_th_pos
        self.right_hand_fingertip_pos_list[1] = self.right_hand_ff_pos
        self.right_hand_fingertip_pos_list[2] = self.right_hand_mf_pos
        self.right_hand_fingertip_pos_list[3] = self.right_hand_rf_pos
        self.right_hand_fingertip_pos_list[4] = self.right_hand_lf_pos

        self.left_hand_fingertip_pos_list[0] = self.left_hand_th_pos
        self.left_hand_fingertip_pos_list[1] = self.left_hand_ff_pos
        self.left_hand_fingertip_pos_list[2] = self.left_hand_mf_pos
        self.left_hand_fingertip_pos_list[3] = self.left_hand_rf_pos
        self.left_hand_fingertip_pos_list[4] = self.left_hand_lf_pos
        
        # generate random values
        self.right_hand_dof_vel_finite_diff = self.dexterous_hand_dof_vel[:, 6:self.num_dexterous_hand_dofs].clone()
        self.left_hand_dof_vel_finite_diff = self.dexterous_hand_another_dof_vel[:, 6:self.num_dexterous_hand_dofs].clone()
        self.right_hand_dof_torque = self.dof_force_tensor[:, 6:self.num_dexterous_hand_dofs].clone()
        self.left_hand_dof_torque = self.dof_force_tensor[:, self.num_dexterous_hand_dofs+6:self.num_dexterous_hand_dofs*2].clone()

        self.right_hand_energy_penalty = ((self.right_hand_dof_torque * self.right_hand_dof_vel_finite_diff).sum(-1)) ** 2
        self.left_hand_energy_penalty = ((self.left_hand_dof_torque * self.left_hand_dof_vel_finite_diff).sum(-1)) ** 2

        contacts = self.contact_tensor.reshape(self.num_envs, -1, 3)  # 39+27
        
        self.object_contacts = contacts[:, self.object_sensor_handles_indices, :]  # 12
        self.object_contacts = torch.norm(self.object_contacts, dim=-1)
        self.object_contacts = torch.where(self.object_contacts >= 0.1, 1.0, 0.0)
        
        self.right_contacts = contacts[:, self.sensor_handle_indices, :]  # 12
        self.right_contacts = torch.norm(self.right_contacts, dim=-1)
        self.right_contacts = torch.where(self.right_contacts >= 0.1, 1.0, 0.0)

        self.left_contacts = contacts[:, self.another_sensor_handle_indices, :]  # 12
        self.left_contacts = torch.norm(self.left_contacts, dim=-1)
        self.left_contacts = torch.where(self.left_contacts >= 0.1, 1.0, 0.0)

        self.all_contact = torch.norm(self.contact_tensor.reshape(self.num_envs, -1, 3), dim=-1)
        self.all_contact = torch.where(self.all_contact >= 0.1, 1.0, 0.0)
        
        rand_floats = torch_rand_float(-1.0, 1.0, (self.num_envs, 63), device=self.device)

        self.compute_sim2real_asymmetric_obs(rand_floats)
        self.compute_sim2real_observation(rand_floats)

    def compute_sim2real_observation(self, rand_floats):
        if self.train_teacher_policy:
            self.obs_buf = self.states_buf.clone()

    def compute_sim2real_asymmetric_obs(self, rand_floats):
        # visualize
        self.states_buf[:, 0 : self.num_dexterous_hand_dofs] = unscale(
            self.dexterous_hand_dof_pos[:, 0 : self.num_dexterous_hand_dofs],
            self.dexterous_hand_dof_lower_limits[0 : self.num_dexterous_hand_dofs],
            self.dexterous_hand_dof_upper_limits[0 : self.num_dexterous_hand_dofs],
        )
        self.states_buf[:, self.num_dexterous_hand_dofs : 2 * self.num_dexterous_hand_dofs] = unscale(
            self.dexterous_hand_another_dof_pos[:, 0 : self.num_dexterous_hand_dofs],
            self.dexterous_hand_dof_lower_limits[0 : self.num_dexterous_hand_dofs],
            self.dexterous_hand_dof_upper_limits[0 : self.num_dexterous_hand_dofs],
        )

        self.states_buf[:, 84:87] = self.dexterous_right_hand_linvel
        self.states_buf[:, 87:90] = self.dexterous_right_hand_angvel
        self.states_buf[:, 90:93] = self.dexterous_left_hand_linvel
        self.states_buf[:, 93:96] = self.dexterous_left_hand_angvel

        self.states_buf[:, 96:99] = self.dexterous_right_hand_pos
        self.states_buf[:, 99:103] = self.dexterous_right_hand_rot

        self.states_buf[:, 103:106] = self.dexterous_left_hand_pos
        self.states_buf[:, 106:110] = self.dexterous_left_hand_rot

        self.states_buf[:, 110:117] = self.object_pose
        self.states_buf[:, 117:120] = self.object_linvel
        self.states_buf[:, 120:123] = self.object_angvel

        self.states_buf[:, 123:126] = self.object_pos - self.obj_params[
            self.seq_idx_tensor, self.progress_buf, 4:7
        ].squeeze(0)
        
        self.states_buf[:, 126:133] = self.object_bottom_pose
        self.states_buf[:, 133:134] = self.object_dof_pos
        self.states_buf[:, 134:141] = self.object_top_pose
        
        skip_frame = 1
        for i in range(10):
            self.states_buf[:, 144 + 22 * i : 147 + 22 * i] = self.obj_params[
                self.seq_idx_tensor, self.progress_buf + i*skip_frame, 4:7
            ].squeeze(0)
            self.states_buf[:, 147 + 22 * i : 151 + 22 * i] = self.obj_rot_quat[
                self.seq_idx_tensor, self.progress_buf + i*skip_frame
            ].squeeze(0)
            self.states_buf[:, 151 + 22 * i : 154 + 22 * i] = self.trans_l[
                self.seq_idx_tensor, self.progress_buf + i*skip_frame
            ].squeeze(0)
            self.states_buf[:, 154 + 22 * i : 158 + 22 * i] = self.rot_l_quat[
                self.seq_idx_tensor, self.progress_buf + i*skip_frame
            ].squeeze(0)
            self.states_buf[:, 158 + 22 * i : 161 + 22 * i] = self.trans_r[
                self.seq_idx_tensor, self.progress_buf + i*skip_frame
            ].squeeze(0)
            self.states_buf[:, 161 + 22 * i : 165 + 22 * i] = self.rot_r_quat[
                self.seq_idx_tensor, self.progress_buf + i*skip_frame
            ].squeeze(0)
            self.states_buf[:, 165 + 22 * i : 166 + 22 * i] = self.obj_params[
                self.seq_idx_tensor, self.progress_buf + i*skip_frame, 0:1
            ].squeeze(0)

        self.index_after_future_frame = 166 + 22 * 9
        
        self.states_buf[:, self.index_after_future_frame:self.index_after_future_frame+1] = self.object_dof - self.obj_params[
            self.seq_idx_tensor, self.progress_buf, 0:1
        ].squeeze(0)
        
        if self.used_hand_type in ["shadow", "schunk", "ability"]:
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 0 : self.index_after_future_frame+1 + 13 * 1] = self.left_hand_ff_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 1 : self.index_after_future_frame+1 + 13 * 2] = self.left_hand_lf_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 2 : self.index_after_future_frame+1 + 13 * 3] = self.left_hand_mf_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 3 : self.index_after_future_frame+1 + 13 * 4] = self.left_hand_rf_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 4 : self.index_after_future_frame+1 + 13 * 5] = self.left_hand_th_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 5 : self.index_after_future_frame+1 + 13 * 6] = self.right_hand_ff_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 6 : self.index_after_future_frame+1 + 13 * 7] = self.right_hand_lf_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 7 : self.index_after_future_frame+1 + 13 * 8] = self.right_hand_mf_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 8 : self.index_after_future_frame+1 + 13 * 9] = self.right_hand_rf_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 9 : self.index_after_future_frame+1 + 13 * 10] = self.right_hand_th_state
            
        elif self.used_hand_type == "allegro":
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 0 : self.index_after_future_frame+1 + 13 * 1] = self.left_hand_ff_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 1 : self.index_after_future_frame+1 + 13 * 2] = self.left_hand_lf_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 3 : self.index_after_future_frame+1 + 13 * 4] = self.left_hand_rf_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 4 : self.index_after_future_frame+1 + 13 * 5] = self.left_hand_th_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 5 : self.index_after_future_frame+1 + 13 * 6] = self.right_hand_ff_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 6 : self.index_after_future_frame+1 + 13 * 7] = self.right_hand_lf_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 8 : self.index_after_future_frame+1 + 13 * 9] = self.right_hand_rf_state
            self.states_buf[:, self.index_after_future_frame+1 + 13 * 9 : self.index_after_future_frame+1 + 13 * 10] = self.right_hand_th_state

        self.states_buf[:, self.index_after_future_frame+1 + 13 * 10: self.index_after_future_frame+1 + 13 * 10 + self.all_contact.shape[1]] = self.all_contact

    def reset(self, env_ids, goal_env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.perturb_direction[env_ids] = torch_rand_float(
            -1, 1, (len(env_ids), 6), device=self.device
        ).squeeze(-1)

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
        
        # reset shadow hand
        pos = self.dexterous_hand_default_dof_pos
        another_pos = self.another_dexterous_hand_default_dof_pos

        self.dexterous_hand_dof_pos[env_ids, :] = pos
        self.dexterous_hand_another_dof_pos[env_ids, :] = another_pos

        self.dexterous_hand_dof_vel[env_ids, :] = self.dexterous_hand_dof_default_vel
        self.dexterous_hand_another_dof_vel[env_ids, :] = self.dexterous_hand_dof_default_vel

        self.prev_targets[env_ids, : self.num_dexterous_hand_dofs] = pos
        self.cur_targets[env_ids, : self.num_dexterous_hand_dofs] = pos

        self.prev_targets[
            env_ids, self.num_dexterous_hand_dofs : self.num_dexterous_hand_dofs * 2
        ] = another_pos
        self.cur_targets[
            env_ids, self.num_dexterous_hand_dofs : self.num_dexterous_hand_dofs * 2
        ] = another_pos

        # reset object
        self.object_dof_pos[env_ids, :] = self.object_default_dof_pos
        self.object_dof_vel[env_ids, :] = torch.zeros_like(self.object_dof_vel[env_ids, :])

        self.prev_targets[env_ids, 2 * self.num_dexterous_hand_dofs :] = self.object_default_dof_pos
        self.cur_targets[env_ids, 2 * self.num_dexterous_hand_dofs :] = self.object_default_dof_pos

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

        self.init_step = 0
        self.end_step_buf[env_ids] = self.init_step + 500
        
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

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_indices),
            len(all_indices),
        )

        # post reset
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.last_actions[env_ids] = torch.zeros_like(self.actions[env_ids])

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

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
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.cur_targets[:, 6 : self.num_dexterous_hand_dofs] = scale(
            self.actions[:, 6:self.num_dexterous_hand_dofs],
            self.dexterous_hand_dof_lower_limits[6 : self.num_dexterous_hand_dofs],
            self.dexterous_hand_dof_upper_limits[6 : self.num_dexterous_hand_dofs],
        )
        self.cur_targets[:, 6 : self.num_dexterous_hand_dofs] = self.act_moving_average * self.cur_targets[:,
                                                                                                    6 : self.num_dexterous_hand_dofs] + (1.0 - self.act_moving_average) * self.prev_targets[:, 6 : self.num_dexterous_hand_dofs]
        self.cur_targets[
            :, self.num_dexterous_hand_dofs + 6 : self.num_dexterous_hand_dofs * 2
        ] = scale(
            self.actions[:, self.num_dexterous_hand_dofs+6:self.num_dexterous_hand_dofs*2],
            self.dexterous_hand_dof_lower_limits[6 : self.num_dexterous_hand_dofs],
            self.dexterous_hand_dof_upper_limits[6 : self.num_dexterous_hand_dofs],
        )
        self.cur_targets[:,self.num_dexterous_hand_dofs + 6 : self.num_dexterous_hand_dofs * 2] = self.act_moving_average * self.cur_targets[:,
                                                                                                    self.num_dexterous_hand_dofs + 6 : self.num_dexterous_hand_dofs * 2] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.num_dexterous_hand_dofs + 6 : self.num_dexterous_hand_dofs * 2]
        
        self.cur_targets[:, self.num_dexterous_hand_dofs*2:self.num_dexterous_hand_dofs*2+1] = self.object_dof_pos

        right_pos_err = (self.r_pos_global - self.dexterous_right_hand_pos)
        right_target_rot = self.r_rot_global
        right_rot_err = orientation_error(right_target_rot, self.dexterous_right_hand_rot)
        right_pos_err += self.actions[:, 0:3] * 0.02
        right_rot_err += self.actions[:, 3:6] * 0.1
        
        if self.use_hierarchy:
            right_pos_err = self.actions[:, 0:3] * 0.04
            right_rot_err = self.actions[:, 3:6] * 0.5
            
        right_dpose = torch.cat([right_pos_err, right_rot_err], -1).unsqueeze(-1)
        right_delta = control_ik(
            self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :6],
            self.device,
            right_dpose,
            self.num_envs,
        )
                
        right_targets = self.dexterous_hand_dof_pos[:, 0:6] + right_delta[:, :6]

        left_pos_err = (self.l_pos_global - self.dexterous_left_hand_pos)
        left_target_rot = self.l_rot_global
        left_rot_err = orientation_error(left_target_rot, self.dexterous_left_hand_rot)
        left_pos_err += self.actions[:, self.num_dexterous_hand_dofs:self.num_dexterous_hand_dofs+3] * 0.02
        left_rot_err += self.actions[:, self.num_dexterous_hand_dofs+3:self.num_dexterous_hand_dofs+6] * 0.1
            
        if self.use_hierarchy:
            left_pos_err = self.actions[:, self.num_dexterous_hand_dofs:self.num_dexterous_hand_dofs+3] * 0.04
            left_rot_err = self.actions[:, self.num_dexterous_hand_dofs+3:self.num_dexterous_hand_dofs+6] * 0.5
            
        left_dpose = torch.cat([left_pos_err, left_rot_err], -1).unsqueeze(-1)
        left_delta = control_ik(
            self.another_jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :6],
            self.device,
            left_dpose,
            self.num_envs,
        )
        left_targets = self.dexterous_hand_another_dof_pos[:, 0:6] + left_delta[:, :6]

        self.cur_targets[:, :6] = right_targets[:, :6].clone()
        self.cur_targets[:, self.num_dexterous_hand_dofs:self.num_dexterous_hand_dofs+6] = left_targets[:, :6].clone()

        self.cur_targets[:, : self.num_dexterous_hand_dofs] = tensor_clamp(
            self.cur_targets[:, 0 : self.num_dexterous_hand_dofs],
            self.dexterous_hand_dof_lower_limits[:],
            self.dexterous_hand_dof_upper_limits[:],
        )

        self.cur_targets[
            :, self.num_dexterous_hand_dofs : self.num_dexterous_hand_dofs * 2
        ] = tensor_clamp(
            self.cur_targets[:, self.num_dexterous_hand_dofs : self.num_dexterous_hand_dofs * 2],
            self.dexterous_hand_dof_lower_limits[:],
            self.dexterous_hand_dof_upper_limits[:],
        )

        self.prev_targets[:, :] = self.cur_targets[:, :].clone()

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        
        ############################################################
        if self.apply_perturbation:
            rand_floats = torch_rand_float(-1.0, 1.0, (self.num_envs, 3), device=self.device)
            self.apply_forces[:, 31*2 + 0, :] = rand_floats * 20
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces), gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)
        #############################################################

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def add_debug_lines(self, env, pos, rot, line_width=1):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

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

@torch.jit.script
def compute_hand_reward(
    rew_buf,
    reset_buf,
    reset_goal_buf,
    progress_buf,
    successes,
    consecutive_successes,
    object_contact,
    left_contact,
    right_contact,
    dexterous_left_hand_pos,
    dexterous_right_hand_pos,
    dexterous_left_hand_rot,
    dexterous_right_hand_rot,
    max_episode_length: float,
    object_pos,
    object_rot,
    target_pos,
    target_rot,
    dexterous_left_hand_dof,
    dexterous_right_hand_dof,
    object_dof,
    trans_r,
    trans_l,
    rot_r_quat,
    rot_l_quat,
    obj_params,
    obj_quat,
    dist_reward_scale: float,
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
    right_hand_energy_penalty,
    left_hand_energy_penalty,
    end_step_buf,
    seq_idx_tensor,
    max_consecutive_successes: int,
    av_factor: float,
    ignore_z_rot: bool,
    object_joint_tolerate: float,
    object_joint_reset: float,
    use_fingertip_reward: int,
    use_hierarchy: int,
    left_fingertip_global,
    right_fingertip_global,
    left_fingertip_pos_list,
    right_fingertip_pos_list,
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
    object_joint_dist = torch.clamp(object_joint_dist - object_joint_tolerate, 0, None)

    left_hand_pos_dist = torch.norm(
        dexterous_left_hand_pos - trans_l[seq_idx_tensor, progress_buf].squeeze(0), p=2, dim=-1
    )
    left_hand_quat_diff = quat_mul(
        dexterous_left_hand_rot, quat_conjugate(rot_l_quat[seq_idx_tensor, progress_buf].squeeze(0))
    )
    left_hand_rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(left_hand_quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    )

    left_hand_pos_dist = torch.clamp(left_hand_pos_dist - 0.15, 0, None)
    left_hand_rot_dist = torch.clamp(left_hand_rot_dist - 0.5, 0, None)

    right_hand_pos_dist = torch.norm(
        dexterous_right_hand_pos - trans_r[seq_idx_tensor, progress_buf].squeeze(0), p=2, dim=-1
    )
    right_hand_quat_diff = quat_mul(
        dexterous_right_hand_rot, quat_conjugate(rot_r_quat[seq_idx_tensor, progress_buf].squeeze(0))
    )
    right_hand_rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(right_hand_quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    )

    right_hand_pos_dist = torch.clamp(right_hand_pos_dist - 0.15, 0, None)
    right_hand_rot_dist = torch.clamp(right_hand_rot_dist - 0.5, 0, None)

    object_reward = 1 * torch.exp(
        -2 * object_rot_dist - 20 * object_pos_dist - 2 * object_joint_dist
    )
    left_hand_reward = 1 * torch.exp(-1 * left_hand_rot_dist - 20 * left_hand_pos_dist)
    right_hand_reward = 1 * torch.exp(-1 * right_hand_rot_dist - 20 * right_hand_pos_dist)

    is_left_contact = (left_contact == 1).any(dim=1, keepdim=False)
    is_right_contact = (right_contact == 1).any(dim=1, keepdim=False)

    jittering_penalty = 0.003 * torch.sum(actions**2, dim=-1)
    energy_penalty = -0.000001 * (right_hand_energy_penalty + left_hand_energy_penalty)
    
    reward = (object_reward + energy_penalty)

    if use_fingertip_reward:
        reward *= torch.exp((torch.norm(left_fingertip_global[:, 0:3] - left_fingertip_pos_list[0], p=2, dim=-1) + torch.norm(left_fingertip_global[:, 3:6] - left_fingertip_pos_list[1], p=2, dim=-1) +
                   torch.norm(left_fingertip_global[:, 6:9] - left_fingertip_pos_list[2], p=2, dim=-1) + torch.norm(left_fingertip_global[:, 9:12] - left_fingertip_pos_list[3], p=2, dim=-1) + 
                   torch.norm(left_fingertip_global[:, 12:15] - left_fingertip_pos_list[4], p=2, dim=-1)) / 5 * (-20))                                                                                                   
        reward *= torch.exp((torch.norm(right_fingertip_global[:, 0:3] - right_fingertip_pos_list[0], p=2, dim=-1) + torch.norm(right_fingertip_global[:, 3:6] - right_fingertip_pos_list[1], p=2, dim=-1) +
                   torch.norm(right_fingertip_global[:, 6:9] - right_fingertip_pos_list[2], p=2, dim=-1) + torch.norm(right_fingertip_global[:, 9:12] - right_fingertip_pos_list[3], p=2, dim=-1) + 
                   torch.norm(right_fingertip_global[:, 12:15] - right_fingertip_pos_list[4], p=2, dim=-1)) / 5 * (-20))
        
    if use_hierarchy:
        reward *= right_hand_reward * left_hand_reward

    # Check env termination conditions, including maximum success number
    resets = torch.where(object_pos[:, 2] <= -10.15, torch.ones_like(reset_buf), reset_buf)

    resets = torch.where(object_pos_dist >= 0.05, torch.ones_like(resets), resets)
    resets = torch.where(object_rot_dist >= 0.5, torch.ones_like(resets), resets)
    resets = torch.where(object_joint_dist >= object_joint_reset, torch.ones_like(resets), resets)
    resets = torch.where(is_left_contact, torch.ones_like(resets), resets)
    resets = torch.where(is_right_contact, torch.ones_like(resets), resets)

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
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def control_ik(j_eef, device: str, dpose, num_envs: int):
    # Set controller parameters
    # IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u

@torch.jit.script
def quaternion_to_rotation_matrix(quaternion):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quaternion (torch.Tensor): Input quaternion tensor with shape (batch_size, 4).
        
    Returns:
        torch.Tensor: Rotation matrix tensor with shape (batch_size, 3, 3).
    """
    q1, q2, q3, q0 = quaternion.unbind(dim=-1)  # Assuming xyzw order
    batch_size = quaternion.size(0)

    # Compute rotation matrix
    rotation_matrix = torch.zeros(batch_size, 3, 3, dtype=quaternion.dtype, device=quaternion.device)
    rotation_matrix[:, 0, 0] = 1 - 2*q2*q2 - 2*q3*q3
    rotation_matrix[:, 0, 1] = 2*q1*q2 - 2*q0*q3
    rotation_matrix[:, 0, 2] = 2*q1*q3 + 2*q0*q2
    rotation_matrix[:, 1, 0] = 2*q1*q2 + 2*q0*q3
    rotation_matrix[:, 1, 1] = 1 - 2*q1*q1 - 2*q3*q3
    rotation_matrix[:, 1, 2] = 2*q2*q3 - 2*q0*q1
    rotation_matrix[:, 2, 0] = 2*q1*q3 - 2*q0*q2
    rotation_matrix[:, 2, 1] = 2*q2*q3 + 2*q0*q1
    rotation_matrix[:, 2, 2] = 1 - 2*q1*q1 - 2*q2*q2

    return rotation_matrix

@torch.jit.script
def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert rotation matrix to quaternion representation of rotation.
    
    Args:
        rotation_matrix (torch.Tensor): Rotation matrix tensor with shape (batch_size, 3, 3).
        
    Returns:
        torch.Tensor: Quaternion representation of rotation with shape (batch_size, 4).
    """
    batch_size = rotation_matrix.size(0)

    # Extract elements from rotation matrix
    r11, r12, r13 = rotation_matrix[:, 0, 0], rotation_matrix[:, 0, 1], rotation_matrix[:, 0, 2]
    r21, r22, r23 = rotation_matrix[:, 1, 0], rotation_matrix[:, 1, 1], rotation_matrix[:, 1, 2]
    r31, r32, r33 = rotation_matrix[:, 2, 0], rotation_matrix[:, 2, 1], rotation_matrix[:, 2, 2]

    # Calculate quaternion components
    qw = torch.sqrt(1.0 + r11 + r22 + r33) / 2.0
    qx = (r32 - r23) / (4.0 * qw)
    qy = (r13 - r31) / (4.0 * qw)
    qz = (r21 - r12) / (4.0 * qw)

    quaternion = torch.stack((qx, qy, qz, qw), dim=-1)
    return quaternion

@torch.jit.script
def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    
    Args:
        q1 (torch.Tensor): First quaternion with shape (batch_size, 4).
        q2 (torch.Tensor): Second quaternion with shape (batch_size, 4).
        
    Returns:
        torch.Tensor: Product quaternion with shape (batch_size, 4).
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((x, y, z, w), dim=-1)
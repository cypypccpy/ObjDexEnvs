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
from utils import o3dviewer

from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
import pickle5
import pickle
import time
import cv2
from collections import deque
# from pointnet2_ops import pointnet2_utils

class ShadowHandFreeWithPhysics(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
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

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)
        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = "block"
        assert self.object_type in ["block", "egg", "pen", "ycb/banana", "ycb/can", "ycb/mug", "ycb/brick"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor1.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "ycb/banana": "urdf/ycb/011_banana/011_banana.urdf",
            "ycb/can": "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
            "ycb/mug": "urdf/ycb/025_mug/025_mug.urdf",
            "ycb/brick": "urdf/ycb/061_foam_brick/061_foam_brick.urdf"
        }

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        # if not (self.obs_type in ["point_cloud", "full_state"]):
        #     raise Exception(
        #         "Unknown type of observations!\nobservationType should be one of: [point_cloud, full_state]")

        print("Obs type:", self.obs_type)

        self.num_point_cloud_feature_dim = 768
        self.num_obs_dict = {
            "full_state": 82
        }

        self.num_hand_obs = 82
        self.up_axis = 'z'

        self.hand_center = ["robot1:palm"]

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 82

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states

        
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 28
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 28

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.enable_camera_sensors = self.cfg["env"]["enableCameraSensors"]
        self.collect_video = False
        self.collect_depth = False
        self.test_depth = False
        self.is_digital_twin = False

        super().__init__(cfg=self.cfg, enable_camera_sensors=self.enable_camera_sensors)

        if self.viewer != None:
            # cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            # cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            cam_pos = gymapi.Vec3(0.5, -0.0, 1.2)
            cam_target = gymapi.Vec3(-0.5, -0.0, 0.2)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.jacobian_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_jacobian_tensor(self.sim, "hand")
        )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.hand_indices[0], "wrist_3_link", gymapi.DOMAIN_ENV
        )

        print("hand_base_rigid_body_index: ", self.hand_base_rigid_body_index)

        self.success_buf = torch.zeros_like(self.rew_buf)
        self.total_steps = 0
        self.envs_list_i = [i for i in range(self.num_envs)]

        self.collect_actions = []
        self.collect_states = []
        self.collect_images = []

        if self.collect_depth:
            import zarr
            zarr_file_path = "/home/user/DexterousHandEnvs/dexteroushandenvs/demonstration/grasp"
            self.root = zarr.open(zarr_file_path, mode='w')

            self.num_student_obs = 19
            self.student_obs_buf = torch.zeros(
            (self.num_envs, self.num_student_obs), device=self.device, dtype=torch.float)

        if self.is_digital_twin:
            import zmq
            context = zmq.Context()
            self.socket = context.socket(zmq.REQ)
            self.socket.connect("tcp://localhost:5555")

        if self.test_depth:
            self.num_student_obs = 19
            self.student_obs_buf = torch.zeros(
            (self.num_envs, self.num_student_obs), device=self.device, dtype=torch.float)

            import robomimic.utils.file_utils as FileUtils
            
            self.policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path="/home/user/robomimic/bc_transformer_trained_models/test/20240729153006/models/model_epoch_2000.pth", device=self.device, verbose=True)
            
            self.depth_image = self.depth_camera_tensors[0]
            self.depth_image = torch.clamp(-self.depth_image, 0, 1.5).unsqueeze(0)

            obs = {"states": self.student_obs_buf[0].cpu().numpy(), "depth_image": self.depth_image.cpu().numpy()}
            
            self.timestep = 0  # always zero regardless of timestep type
            self.update_obs(obs, reset=True)
            self.obs_history = self._get_initial_obs_history(init_obs=obs)
            self.obs = self._get_stacked_obs_from_history()

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"
        shadow_hand_asset_file = "urdf/shadow_hand/shadow_hand_right_free.urdf"

        object_asset_file = self.asset_files_dict[self.object_type]

        # load shadow hand asset
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
        # asset_options.vhacd_params.resolution = 100000

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        shadow_hand_dof_stiffness_list = [200, 200, 100, 100, 50, 50, 50]
        shadow_hand_dof_damping_list = [20, 20, 10, 10, 10, 5, 5]
        shadow_hand_dof_effort_list = [60, 60, 30, 30, 10, 10, 10]
        shadow_hand_dof_velocity_list = [1, 1, 1, 1, 1, 1, 1]

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

            shadow_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if i < 6:
                shadow_hand_dof_props['stiffness'][i] = shadow_hand_dof_stiffness_list[i]
                shadow_hand_dof_props['effort'][i] = shadow_hand_dof_effort_list[i]
                shadow_hand_dof_props['damping'][i] = shadow_hand_dof_damping_list[i]
                shadow_hand_dof_props['velocity'][i] = shadow_hand_dof_velocity_list[i]

        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        self.file_paths = []
        self.is_multi_object = False
        self.point_cloud_datas = []
        self.num_point_downsample = 200

        if self.is_multi_object:
            taochen_directory = "/home/user/DexterousHandEnvs/assets/urdf/taochen_assets/miscnet"
            # Walk through the directory's direct subdirectories
            for subdir in os.listdir(taochen_directory):
                subdir_path = os.path.join(taochen_directory, subdir)
                if os.path.isdir(subdir_path):
                    for root, _, files in os.walk(subdir_path):
                        for file in files:
                            if file.endswith('.urdf'):
                                # Construct file path
                                file_path = os.path.join(root, file)
                                # Add to list
                                self.file_paths.append(file_path)

            for i, file_path in enumerate(self.file_paths):
                # Load the mesh
                with open(file_path.split("model.urd")[0] + "point_cloud_200_pts.pkl", "rb") as f:
                    point_cloud_data = pickle5.load(f)

                # Convert numpy array to torch tensor
                point_cloud_tensor = torch.tensor(point_cloud_data, device=self.device, dtype=torch.float32)

                self.point_cloud_datas.append(point_cloud_tensor)

        else:
            self.file_paths.append("/home/user/DexterousHandEnvs/assets/urdf/objects/cube_multicolor2.urdf")
            with open("/home/user/DexterousHandEnvs/assets/urdf/objects/point_cloud_200_pts.pkl", "rb") as f:
                point_cloud_data = pickle5.load(f)

            point_cloud_tensor = torch.tensor(point_cloud_data, device=self.device, dtype=torch.float32)
            self.point_cloud_datas.append(point_cloud_tensor)

        # load manipulated object and goal assets
        self.object_asset_dict = {
            "object": "IKEA/table_top.urdf",
        }
        print("ENTER ASSET CREATING!")
        self.object_names = []
        self.object_assets = []
        self.object_start_poses = []

        self.object_names.append("object")

        for i, file_path in enumerate(self.file_paths):
            object_asset_file = file_path.split("DexterousHandEnvs/assets/")[1]
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 1000
            object_asset_options.flip_visual_attachments = False
            object_asset_options.fix_base_link = False    
            object_asset_options.collapse_fixed_joints = True
            object_asset_options.disable_gravity = False
            object_asset_options.thickness = 0.001
            object_asset_options.angular_damping = 0.01
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 1000000
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

            object_start_pose = gymapi.Transform()
            object_start_pose.p = gymapi.Vec3(-0.0, -0.0, 0.8)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(-1.571, 0, 0)

            self.object_assets.append(object_asset)
            self.object_start_poses.append(object_start_pose)

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0.2, 0.2, 1.0)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 1.571, 3.1415)

        table_dims = gymapi.Vec3(1.5, 1.5, 0.75)
        # table_dims = gymapi.Vec3(1.5, 1.5, 0.75)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.density = 1000
        table_asset_options.flip_visual_attachments = False
        table_asset_options.collapse_fixed_joints = True
        # table_asset_options.fix_base_link = True
        table_asset_options.disable_gravity = False
        table_asset_options.thickness = 0.001
        table_asset_options.angular_damping = 0.01
        table_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        table_asset_options.override_com = True
        table_asset_options.override_inertia = True
        table_asset_options.vhacd_enabled = True
        table_asset_options.vhacd_params = gymapi.VhacdParams()
        table_asset_options.vhacd_params.resolution = 100000

        table_asset = self.gym.create_box(
            self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options
        )

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0, 0, table_dims.z / 2)
        table_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies * 2 + 200
        max_agg_shapes = self.num_shadow_hand_shapes * 2 + 200

        self.shadow_hands = []
        self.envs = []

        self.cameras_0 = []
        self.camera_tensors_0 = []
        self.depth_camera_tensors_0 = []
        self.cameras_1 = []
        self.camera_tensors_1 = []
        self.depth_camera_tensors_1 = []
        self.cameras_2 = []
        self.camera_tensors_2 = []
        self.depth_camera_tensors_2 = []
        self.cameras_3 = []
        self.camera_tensors_3 = []
        self.depth_camera_tensors_3 = []
        self.camera_view_matrixs = []
        self.camera_proj_matrixs = []

        self.hand_start_states = []

        self.hand_indices = []
        self.another_hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []

        self.pts_states = []

        for obj_i, name in enumerate(self.object_names):
            setattr(self, f"{name}_indices", [])
            setattr(self, f"{name}_init_state", [])

        if self.enable_camera_sensors:
            self.camera_tensors_list = []
            self.depth_camera_tensors_list = []
            self.cameras = []
            self.camera_tensors = []
            self.camera_view_matrixs = []
            self.camera_proj_matrixs = []

            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 160
            self.camera_props.height = 120
            # self.camera_props.width = 1280
            # self.camera_props.height = 800
            self.camera_props.enable_tensors = True

            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

            self.camera_v2, self.camera_u2 = torch.meshgrid(
                self.camera_v, self.camera_u, indexing='ij'
            )

            if self.collect_video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                
                self.video_out_list = []
                for i in range(self.num_envs):
                    save_video_dir = '/home/user/DexterousHandEnvs/dexteroushandenvs/videos/grasp/'
                    if not os.path.exists(save_video_dir):
                        os.makedirs(save_video_dir)
                        
                    self.out = cv2.VideoWriter(save_video_dir + '{}.mp4'.format(i), fourcc, 30.0, (1280, 800))
                    self.video_out_list.append(self.out)

        if self.point_cloud_debug:
            import open3d as o3d
            from utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()
        else :
            self.pointCloudVisualizer = None

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, 0, 0)
            
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add object
            assets_i = i % len(self.object_assets)
            self.pts_states.append(self.point_cloud_datas[assets_i])

            for obj_i, _ in enumerate(self.object_names):
                object_handle = self.gym.create_actor(env_ptr, self.object_assets[assets_i], self.object_start_poses[obj_i], self.object_names[obj_i], i, 0, 0)
                eval("self.{}_init_state".format(self.object_names[obj_i])).append([self.object_start_poses[obj_i].p.x, self.object_start_poses[obj_i].p.y, self.object_start_poses[obj_i].p.z,
                                            self.object_start_poses[obj_i].r.x, self.object_start_poses[obj_i].r.y, self.object_start_poses[obj_i].r.z, self.object_start_poses[obj_i].r.w,
                                            0, 0, 0, 0, 0, 0])
                object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
                eval("self.{}_indices".format(self.object_names[obj_i])).append(object_idx)

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.9, 0.8)
            )
            self.table_indices.append(table_idx)

            if self.enable_camera_sensors:
                camera_handle_0 = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle_0, env_ptr, gymapi.Vec3(0.1, 0, 0.9), gymapi.Vec3(-0.1, 0, 0.8))
                camera_tensor_0 = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle_0, gymapi.IMAGE_COLOR)
                depth_camera_tensor_0 = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle_0, gymapi.IMAGE_DEPTH)
                torch_cam_tensor_0 = gymtorch.wrap_tensor(camera_tensor_0)
                depth_torch_cam_tensor_0 = gymtorch.wrap_tensor(depth_camera_tensor_0)

                camera_handle_1 = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle_1, env_ptr, gymapi.Vec3(-0.1, 0, 0.9), gymapi.Vec3(0.1, 0, 0.8))
                camera_tensor_1 = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle_1, gymapi.IMAGE_COLOR)
                depth_camera_tensor_1 = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle_1, gymapi.IMAGE_DEPTH)
                torch_cam_tensor_1 = gymtorch.wrap_tensor(camera_tensor_1)
                depth_torch_cam_tensor_1 = gymtorch.wrap_tensor(depth_camera_tensor_1)

                camera_handle_2 = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle_2, env_ptr, gymapi.Vec3(0.0, 0.1, 0.9), gymapi.Vec3(-0.0, -0.1, 0.8))
                camera_tensor_2 = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle_2, gymapi.IMAGE_COLOR)
                depth_camera_tensor_2 = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle_2, gymapi.IMAGE_DEPTH)
                torch_cam_tensor_2 = gymtorch.wrap_tensor(camera_tensor_2)
                depth_torch_cam_tensor_2 = gymtorch.wrap_tensor(depth_camera_tensor_2)

                camera_handle_3 = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle_3, env_ptr, gymapi.Vec3(0.0, -0.1, 0.9), gymapi.Vec3(-0., 0.1, 0.8))
                camera_tensor_3 = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle_3, gymapi.IMAGE_COLOR)
                depth_camera_tensor_3 = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle_3, gymapi.IMAGE_DEPTH)
                torch_cam_tensor_3 = gymtorch.wrap_tensor(camera_tensor_3)
                depth_torch_cam_tensor_3 = gymtorch.wrap_tensor(depth_camera_tensor_3)
                # cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                # cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            if self.enable_camera_sensors:
                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors_0.append(torch_cam_tensor_0)
                self.depth_camera_tensors_0.append(depth_torch_cam_tensor_0)
                self.cameras_0.append(camera_handle_0)
                self.camera_tensors_1.append(torch_cam_tensor_1)
                self.depth_camera_tensors_1.append(depth_torch_cam_tensor_1)
                self.cameras_1.append(camera_handle_1)
                self.camera_tensors_2.append(torch_cam_tensor_2)
                self.depth_camera_tensors_2.append(depth_torch_cam_tensor_2)
                self.cameras_2.append(camera_handle_2)
                self.camera_tensors_3.append(torch_cam_tensor_3)
                self.depth_camera_tensors_3.append(depth_torch_cam_tensor_3)
                self.cameras_3.append(camera_handle_3)
                # self.camera_view_matrixs.append(cam_vinv)
                # self.camera_proj_matrixs.append(cam_proj)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        self.camera_tensors_list.append(self.camera_tensors_0)
        self.camera_tensors_list.append(self.camera_tensors_1)
        self.camera_tensors_list.append(self.camera_tensors_2)
        self.camera_tensors_list.append(self.camera_tensors_3)
        self.depth_camera_tensors_list.append(self.depth_camera_tensors_0)
        self.depth_camera_tensors_list.append(self.depth_camera_tensors_1)
        self.depth_camera_tensors_list.append(self.depth_camera_tensors_2)
        self.depth_camera_tensors_list.append(self.depth_camera_tensors_3)

        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices = to_torch(self.another_hand_indices, dtype=torch.long, device=self.device)

        for obj_i, name in enumerate(self.object_names):
            init_state = getattr(self, f"{name}_init_state")
            new_init_state = to_torch(init_state, device=self.device).view(self.num_envs, 13)
            setattr(self, f"{name}_init_state", new_init_state)

            indices = getattr(self, f"{name}_indices")
            new_indices = to_torch(indices, dtype=torch.long, device=self.device)
            setattr(self, f"{name}_indices", new_indices)

        self.goal_states = self.object_init_state.clone()
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

        self.pts_states = torch.stack(
            self.pts_states
        ).view(self.num_envs, self.num_point_downsample, 3)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_states, self.goal_pos, self.goal_rot, self.shadow_right_hand_palm_pos, self.shadow_right_hand_rot, self.shadow_right_hand_pos, self.distance_features, self.middle_point,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        self.total_steps += 1

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.obs_type in ["point_cloud", "point_cloud_for_distill"]:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        for obj_i, name in enumerate(self.object_names):
            setattr(self, f"{name}_states", self.root_state_tensor[getattr(self, f"{name}_indices"), 0:13])
            setattr(self, f"{name}_pose", self.root_state_tensor[getattr(self, f"{name}_indices"), 0:7])
            setattr(self, f"{name}_pos", self.root_state_tensor[getattr(self, f"{name}_indices"), 0:3])
            setattr(self, f"{name}_rot", self.root_state_tensor[getattr(self, f"{name}_indices"), 3:7])
            setattr(self, f"{name}_linvel", self.root_state_tensor[getattr(self, f"{name}_indices"), 7:10])
            setattr(self, f"{name}_angvel", self.root_state_tensor[getattr(self, f"{name}_indices"), 10:13])

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.shadow_right_hand_pos = self.rigid_body_states[:, 7, 0:3]
        self.shadow_right_hand_rot = self.rigid_body_states[:, 7, 3:7]
        self.shadow_right_hand_linvel = self.rigid_body_states[:, 7, 7:10]
        self.shadow_right_hand_angvel = self.rigid_body_states[:, 7, 10:13]

        self.shadow_right_hand_palm_pos = self.shadow_right_hand_pos
        
        # self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        # self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        # self.fingertip_another_state = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:13]
        # self.fingertip_another_pos = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:3]
        self.pts_states_rotated = quat_apply(self.object_rot.unsqueeze(1).repeat(1, self.num_point_downsample, 1), self.pts_states)
        self.distance_features, self.closest_vertices = compute_distance_features(self.pts_states_rotated + self.object_pos.unsqueeze(1), self.rigid_body_states[:, 10:21, 0:3])

        self.right_hand_index_fingertip_pos = self.rigid_body_states[:, 15, 0:3]
        self.right_hand_index_fingertip_pos = self.right_hand_index_fingertip_pos + quat_apply(self.rigid_body_states[:, 15, 3:7], to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.03)
        self.middle_point = (self.rigid_body_states[:, 13, 0:3] + self.right_hand_index_fingertip_pos) / 2

        if self.enable_camera_sensors:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            camera_rgb_image_0 = self.camera_rgb_visulization(self.camera_tensors_list[0], env_id=0, is_depth_image=False)
            camera_rgb_image_1 = self.camera_rgb_visulization(self.camera_tensors_list[1], env_id=0, is_depth_image=False)
            camera_rgb_image_2 = self.camera_rgb_visulization(self.camera_tensors_list[2], env_id=0, is_depth_image=False)
            camera_rgb_image_3 = self.camera_rgb_visulization(self.camera_tensors_list[3], env_id=0, is_depth_image=False)
            camera_depth_image_0 = self.camera_rgb_visulization(self.depth_camera_tensors_list[0], env_id=0, is_depth_image=True)
            camera_depth_image_1 = self.camera_rgb_visulization(self.depth_camera_tensors_list[1], env_id=0, is_depth_image=True)
            camera_depth_image_2 = self.camera_rgb_visulization(self.depth_camera_tensors_list[2], env_id=0, is_depth_image=True)
            camera_depth_image_3 = self.camera_rgb_visulization(self.depth_camera_tensors_list[3], env_id=0, is_depth_image=True)

            images = [
                camera_rgb_image_0, camera_rgb_image_1, camera_rgb_image_2, camera_rgb_image_3,
                self.convert_to_rgb(camera_depth_image_0), self.convert_to_rgb(camera_depth_image_1),
                self.convert_to_rgb(camera_depth_image_2), self.convert_to_rgb(camera_depth_image_3)
            ]

            # Resize images to the same size if needed
            # Assuming all images are of the same size, otherwise, resize them to the same size

            # Concatenate images horizontally to form two rows
            row1 = np.hstack(images[:4])
            row2 = np.hstack(images[4:])

            # Concatenate rows vertically to form the final 2x4 grid
            grid_image = np.vstack([row1, row2])

            cv2.imshow("DEBUG_VIS", grid_image)
            cv2.waitKey(1)

            self.gym.end_access_image_tensors(self.sim)

        self.compute_states()

        if self.collect_depth or self.test_depth:
            self.compute_student_obs()

    def compute_student_obs(self):
        # visualize
        self.student_obs_buf[:, : self.num_shadow_hand_dofs] = unscale(
            self.shadow_hand_dof_pos[:, : self.num_shadow_hand_dofs],
            self.shadow_hand_dof_lower_limits[ : self.num_shadow_hand_dofs],
            self.shadow_hand_dof_upper_limits[ : self.num_shadow_hand_dofs],
        )

        # self.student_obs_buf[:, 19:22] = self.shadow_right_hand_pos
        # self.student_obs_buf[:, 25:29] = self.shadow_right_hand_rot

        # self.student_obs_buf[:, 29:36] = self.object_states[:, 0:7]

    def compute_states(self):
        # visualize
        self.states_buf[:, : self.num_shadow_hand_dofs] = unscale(
            self.shadow_hand_dof_pos[:, : self.num_shadow_hand_dofs],
            self.shadow_hand_dof_lower_limits[ : self.num_shadow_hand_dofs],
            self.shadow_hand_dof_upper_limits[ : self.num_shadow_hand_dofs],
        )

        self.states_buf[:, 13:16] = self.shadow_right_hand_linvel
        self.states_buf[:, 16:19] = self.shadow_right_hand_angvel

        self.states_buf[:, 19:22] = self.middle_point - self.object_states[:, 0:3]
        self.states_buf[:, 22:25] = self.middle_point

        self.states_buf[:, 25:29] = self.shadow_right_hand_rot

        self.states_buf[:, 29:36] = self.object_states[:, 0:7]
        self.states_buf[:, 36:39] = self.object_states[:, 7:10]
        self.states_buf[:, 39:42] = self.object_states[:, 10:13]

        self.states_buf[:, 42:49] = self.goal_pose

        self.states_buf[:, 49:49+11*3] = self.distance_features.view(self.num_envs, -1)

        self.obs_buf = self.states_buf.clone()

    def calc_succ_rate(self, env_ids):
        self.success_buf[env_ids] = torch.where(
            self.root_state_tensor[self.object_indices, 0:3][env_ids, 2] > 0.85,
            torch.ones_like(self.success_buf[env_ids]),
            torch.zeros_like(self.success_buf[env_ids])
        )

        print("success_rate: ", self.success_buf[:].mean())

    def reset(self, env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 12), device=self.device)

        self.calc_succ_rate(env_ids)

        # reset object
        # for obj_i, name in enumerate(self.object_names):
        #     indices_attr = f"{name}_indices"
        #     init_state_attr = f"{name}_init_state"

        #     self.root_state_tensor[getattr(self, indices_attr)[env_ids]] = getattr(self, init_state_attr)[env_ids].clone()

        #     quat = quat_from_euler_xyz(torch.ones_like(rand_floats[:, obj_i*0]) * -1.571, torch.ones_like(rand_floats[:, obj_i*1]) * 0, rand_floats[:, obj_i*2] * 3.1415)
        #     # self.root_state_tensor[getattr(self, indices_attr)[env_ids], 0] = self.root_state_tensor[getattr(self, indices_attr)[env_ids], 0] + rand_floats[:, obj_i*3] * 0.0
        #     # self.root_state_tensor[getattr(self, indices_attr)[env_ids], 1] = self.root_state_tensor[getattr(self, indices_attr)[env_ids], 1] + rand_floats[:, obj_i*4] * 0.0
        #     self.root_state_tensor[getattr(self, indices_attr)[env_ids], 0] = self.root_state_tensor[getattr(self, indices_attr)[env_ids], 0] + rand_floats[:, obj_i*3] * 0.15
        #     self.root_state_tensor[getattr(self, indices_attr)[env_ids], 1] = self.root_state_tensor[getattr(self, indices_attr)[env_ids], 1] + rand_floats[:, obj_i*4] * 0.15
        #     self.root_state_tensor[getattr(self, indices_attr)[env_ids], 3:7] = quat[:, 0:4]

        # reset goal
        self.goal_states[env_ids] = self.root_state_tensor[self.object_indices[env_ids]]
        self.goal_states[env_ids, 2] += 0.2

        object_indices = torch.unique(
            torch.cat(
                [
                    self.table_indices[env_ids],  # Add this directly
                    *[getattr(self, f"{name}_indices")[env_ids] for name in self.object_names]
                ]
            ).to(torch.int32)
        )
        # reset shadow hand
        reset_noise = torch.rand((len(env_ids), 24), device=self.device)

        pos = self.shadow_hand_default_dof_pos

        self.shadow_hand_dof_pos[env_ids, :] = pos

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(hand_indices))  

        all_indices = torch.unique(torch.cat([hand_indices,
                                                 object_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(hand_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.actions = actions.clone().to(self.device)
        if self.test_depth:
            self.act = self.policy(ob=self.obs)

            self.actions = to_torch(self.act[:], device=self.device).unsqueeze(0)

        self.cur_targets[:, 6:28] = scale(self.actions[:, 6:28],
                                                                self.shadow_hand_dof_lower_limits[6:28], self.shadow_hand_dof_upper_limits[6:28])
        
        self.cur_targets[:, 0:6] = torch.zeros_like(self.prev_targets[:, 0:6])
        

        self.prev_targets[:] = self.cur_targets[:].clone()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.test_depth:
            obs = {"states": self.student_obs_buf[0].cpu().numpy(), "depth_image": self.depth_image.cpu().numpy()}

            self.update_obs(obs, action=self.act, reset=False)
            
            # for k in self.obs_history:
            #     print(self.obs_history[k])
            # exit()
            for k in obs:
                self.obs_history[k].append(obs[k][None])
            self.obs = self._get_stacked_obs_from_history()

        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # for i in range(11):
        #     p0 = self.rigid_body_states[0, i+9, 0:3].cpu().numpy()
        #     posy = self.closest_vertices[0, i].cpu().numpy()
        #     self.gym.add_lines(
        #                 self.viewer,
        #                 self.envs[0],
        #                 2,
        #                 [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]],
        #                 [0.85, 0.1, 0.1],
        #     )
        # for i in range(200):
        #     self.add_debug_lines(self.envs[0], self.pts_states_rotated[0, i, :3] + self.object_pos[0], self.object_rot[0], line_width=2)
        # self.add_debug_lines(self.envs[0], self.middle_point[0, :3], self.object_rot[0], line_width=2)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

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
    def camera_rgb_visulization(self, camera_tensors, env_id=0, is_depth_image=False):
        if not is_depth_image:
            torch_rgba_tensor = camera_tensors[env_id].clone()
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

        if is_depth_image:
            torch_depth_tensor = camera_tensors[env_id].clone()
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -1, 1).cpu().numpy()
            # torch_depth_tensor = augment_depth_image(torch_depth_tensor)
            torch_depth_tensor = (torch_depth_tensor + 1) / 2.0
            # torch_depth_tensor = (torch_depth_tensor - torch_depth_tensor.min()) / (torch_depth_tensor.max() - torch_depth_tensor.min())

            camera_image = torch_depth_tensor
            camera_image = np.uint8(camera_image * 255)

        return camera_image

    # Convert depth images to 3-channel images
    def convert_to_rgb(self, depth_image):
        # Convert single-channel depth image to 3-channel
        return cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)

    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

    def sample_points(self, points, sample_num=1000, sample_mathed='furthest'):
        eff_points = points[points[:, 2]>0.04]
        if eff_points.shape[0] < sample_num :
            eff_points = points
        if sample_mathed == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_mathed == 'furthest':
            sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape), sample_num)
            sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points

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
            obs_history[k] = deque(
                [init_obs[k][None] for _ in range(10)], 
                maxlen=10,
            )
        return obs_history

    def update_obs(self, obs, action=None, reset=False):
        obs["timesteps"] = np.array([self.timestep])
        
        if reset:
            obs["actions"] = np.zeros(self.num_actions)
        else:
            self.timestep += 1
            obs["actions"] = action[:self.num_actions]

    def _get_stacked_obs_from_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a 
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        return { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }


#####################################################################
###=========================jit functions=========================###
#####################################################################

def map_range(value, in_min, in_max, out_min, out_max):
    # Calculate the slope (a) and y-intercept (b) of the linear mapping
    slope = (out_max - out_min) / (in_max - in_min)
    intercept = out_min - slope * in_min
    
    # Apply the linear mapping
    mapped_value = slope * value + intercept
    
    return mapped_value

@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    # time1 = time.time()
    depth_buffer = camera_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = camera_view_matrix_inv

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_states, target_pos, target_rot, right_hand_pos, right_hand_rot, right_hand_wrist_pos, distance_features, middle_point,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_states[:, 0:3], p=2, dim=-1)
    # goal_dist = torch.abs(target_pos[:, 2] - object_states[:, 2])

    # Orientation alignment for the cube in hand and goal cube
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    goal_dist_rew = 5 * (0.2 - goal_dist)

    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # print(torch.norm(distance_features[:, :, 0:3] - object_states[:, 0:3].unsqueeze(1), p=2, dim=-1, keepdim=False).shape)
    right_hand_to_object_dist = torch.sum(torch.norm(distance_features[:, :, 0:3], p=2, dim=-1, keepdim=False), dim=1) / 20
    right_hand_to_object_dist = torch.clamp(right_hand_to_object_dist - 0.04, 0, None)

    middle_point_to_object_dist = torch.norm(middle_point - object_states[:, 0:3], p=2, dim=-1)
    middle_point_to_object_dist = torch.clamp(middle_point_to_object_dist - 0.04, 0, None)

    middle_point_reward = torch.exp(-0.2*(middle_point_to_object_dist * dist_reward_scale))

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # hand_over_reward = torch.where(object_pos[:, 0] > 0, 
    #                                 torch.exp(-0.2*(left_hand_to_object_dist * dist_reward_scale)) + torch.exp(-0.2*(right_hand_to_object_dist * dist_reward_scale)), torch.exp(-0.2*(right_hand_to_object_dist * dist_reward_scale)))

    hand_over_reward = torch.exp(-0.2*(right_hand_to_object_dist * dist_reward_scale))
    # hand_over_reward = torch.exp(-0.1*(right_hand_to_object_dist * dist_reward_scale))

    reward = 10 * goal_dist_rew + hand_over_reward + 2 * middle_point_reward
    # reward = 10 * goal_dist_rew + middle_point_reward

    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), reset_buf)

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(goal_dist) <= 0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = torch.where(successes == 0, 
                    torch.where(goal_dist < 0.03, torch.ones_like(successes), successes), successes)

    # Success bonus: orientation is within `success_tolerance` of goal orientation

    # Fall penalty: distance to the goal is larger than a threashold
    # reward = torch.where(object_pos[:, 2] <= 0.2, reward + fall_penalty, reward)

    # Apply penalty for not reaching the goal
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes).mean()

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(j_eef, device: str, dpose, num_envs):
    # Set controller parameters
    # IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u

def compute_distance_features(object_vertices, hand_link_positions):
    # 提取可抓握部分的顶点
    graspable_vertices = object_vertices[:, :, :]  # (B, G, 3)
    
    # 获取 batch size 和 link 数量
    B, L, _ = hand_link_positions.shape
    _, G, _ = graspable_vertices.shape
    
    # 扩展维度以进行广播
    hand_link_positions_expanded = hand_link_positions.unsqueeze(2)  # (B, L, 1, 3)
    graspable_vertices_expanded = graspable_vertices.unsqueeze(1)  # (B, 1, G, 3)
    
    # 计算每个 hand link 到每个 graspable vertex 的距离
    distances = torch.norm(hand_link_positions_expanded - graspable_vertices_expanded, dim=-1)  # (B, L, G)
    
    # 查找最近的 graspable vertex
    min_distances, min_indices = torch.min(distances, dim=-1)  # (B, L)
    
    # 获取最小距离对应的顶点坐标
    closest_vertices = torch.gather(graspable_vertices, 1, min_indices.unsqueeze(-1).expand(-1, -1, 3))  # (B, L, 3)
    
    # 计算差向量
    difference_vectors = hand_link_positions - closest_vertices  # (B, L, 3)
    
    return difference_vectors, closest_vertices

def augment_depth_image(depth_img):
    # Constants for probabilities
    P_dropout = 0.001
    P_randu = 0.001
    P_stick = 0.0005

    h, w = depth_img.shape
    max_length = 18  # Maximum length of the linear segment
    max_width = 3  # Width of the linear segment

    # Deep copy of the input depth image
    augmented_img = depth_img.copy()

    # Augmentation 1: Dropout pixels
    mask_dropout = np.random.rand(*depth_img.shape) < P_dropout
    augmented_img[mask_dropout] = 0

    # Augmentation 2: Random values in range (-0.5, -1.3)
    mask_randu = np.random.rand(*depth_img.shape) < P_randu
    rand_values = np.random.uniform(-1.3, -0.5, size=depth_img.shape)
    augmented_img[mask_randu] = rand_values[mask_randu]

    # Augmentation 3: Linear segments (stick artifacts)
    mask = np.random.random((h, w)) < P_stick

    # Step 2: Define segment parameters
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1:
                # Step 3: Apply linear segment
                direction = np.random.uniform(0, 2*np.pi)  # Random angle in radians
                length = np.random.randint(1, max_length + 1)
                width = np.random.randint(1, max_width + 1)

                depth = np.random.uniform(-1.3, -0.5)
                
                # Calculate end point based on direction and length
                end_x = int(round(j + length * np.cos(direction)))
                end_y = int(round(i + length * np.sin(direction)))
                
                # Step 4: Ensure segment stays within bounds
                if end_x < 0 or end_x >= w or end_y < 0 or end_y >= h:
                    continue  # Skip if end point is out of bounds
                
                # Step 5: Draw the stick by setting pixel values
                if width == 1:
                    # Single-pixel width stick
                    rr, cc = line(j, i, end_x, end_y)
                    rr = np.clip(rr, 0, h - 1)
                    cc = np.clip(cc, 0, w - 1)
                    augmented_img[rr, cc] = depth

                else:
                    # Wider stick
                    half_width = width // 2
                    for k in range(-half_width, half_width + 1):
                        start_point = (j + k, i)
                        end_point = (end_x + k, end_y)
                        rr, cc = line(start_point[0], start_point[1], end_point[0], end_point[1])
                        rr = np.clip(rr, 0, h - 1)
                        cc = np.clip(cc, 0, w - 1)
                        augmented_img[rr, cc] = depth

    return augmented_img

# Helper function to generate line coordinates
def line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return np.array([p[1] for p in points]), np.array([p[0] for p in points])

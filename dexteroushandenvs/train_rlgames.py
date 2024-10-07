# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import numpy as np
import random

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_sarl import *
import os

from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

import yaml
import gym
# from utils.rl_games_custom import
from rl_games.common.algo_observer import IsaacAlgoObserver

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__ == '__main__':
    set_np_formatting()
    args = get_args(use_rlg_config=True)
    if args.checkpoint == "Base":
        args.checkpoint = ""

    config_name = "cfg/arctic/arctic.yaml"

    args.task_type = "RLgames"
    print('Loading config: ', config_name)

    args.cfg_train = config_name
    cfg, cfg_train, logdir = load_cfg(args, use_rlg_config=True)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    cfg_train["seed"] = args.seed
    cfg["env"]["numEnvs"] = args.num_envs
    cfg["env"]["test"] = args.play
    if args.object:
        cfg["env"]["used_training_objects"] = [args.object]
    if args.hand:
        cfg["env"]["used_hand_type"] = args.hand
        
    if args.enable_camera:
        cfg["env"]["enableCameraSensors"] = True

    if args.traj_index:
        cfg["env"]["traj_index"] = args.traj_index

    if args.traj_index:
        cfg["env"]["traj_index"] = args.traj_index

    if args.use_fingertip_reward:
        cfg["env"]["use_fingertip_reward"] = args.use_fingertip_reward
        
    if args.use_hierarchy:
        cfg["env"]["use_hierarchy"] = args.use_hierarchy

    cfg["record_video"] = False
            
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))

    cfg["experiment_name"] = args.task
    if cfg["env"]["used_training_objects"]:
        for i in cfg["env"]["used_training_objects"]:
            cfg["experiment_name"] += "_{}".format(i)
            cfg["experiment_name"] += "_{}".format(args.hand)
            cfg["experiment_name"] += "_{}".format(args.traj_index)
            if args.use_fingertip_reward:
                cfg["experiment_name"] += "_use_fingertip_reward"
            if args.use_hierarchy:
                cfg["experiment_name"] += "_use_hierarchy"
                
    agent_index = None
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    # override
    with open(config_name, 'r') as stream:
        rlgames_cfg = yaml.safe_load(stream)
        rlgames_cfg['params']['config']['name'] = cfg["experiment_name"]
        rlgames_cfg['params']['config']['num_actors'] = env.num_environments
        rlgames_cfg['params']['seed'] = cfg_train["seed"]
        rlgames_cfg['params']['config']['env_config']['seed'] = cfg_train["seed"]
        rlgames_cfg['params']['config']['vec_env'] = env
        rlgames_cfg['params']['config']['env_info'] = env.get_env_info()
        rlgames_cfg['params']['config']['device'] = args.rl_device
    
    
    vargs = vars(args)
    algo_observer = IsaacAlgoObserver()

    runner = Runner(algo_observer)

    runner.load(rlgames_cfg)
    runner.reset()
    
    runner.run(vargs)

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from tasks.dexterous_hand_arctic import DexterousHandArctic
from tasks.hand_base.vec_task_rlgames import RLgamesVecTaskPython

from utils.config import warn_task_name

import json
import gym


def parse_task(args, cfg, cfg_train, sim_params, agent_index):
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    if args.task_type == "RLgames":
        print("Task type: RLgames")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=False,
            )
        except NameError as e:
            print(e)
            warn_task_name()

        if args.record_video:
            if args.record_video_interval:
                record_video_interval = int(args.record_video_interval)
            else:
                record_video_interval = int(1)
            task.is_vector_env = True
            from datetime import datetime
            now = datetime.now()
            args.save_time_stamp = now.strftime("%Y%m%d%H%M%S")

            if args.play:
                task = gym.wrappers.RecordVideo(task, f"{args.record_video_path}/{args.task}/",\
                        # step_trigger=lambda step: step % record_video_interval == 0, # record the videos every record_video_interval steps
                        episode_trigger=lambda episode: episode % record_video_interval == 0, # record the videos every record_video_interval episodes
                        # video_length=record_video_length,
                        name_prefix = f"{args.task}_{args.algo}_{args.save_time_stamp}_video"
                        )

        env = RLgamesVecTaskPython(task, rl_device)
    return task, env
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append('/home/user/DexterousHandEnvs/dexteroushandenvs/algorithms/td_mpc2/tdmpc2/tdmpc2')

def load_config(config_name: str, config_path: str = '/home/user/DexterousHandEnvs/dexteroushandenvs/algorithms/td_mpc2/tdmpc2/tdmpc2/config.yaml') -> dict:
    # Initialize Hydra
    hydra.initialize(config_path=config_path)
    # Compose the configuration
    cfg = hydra.compose(config_name=config_name)
    # Convert to a dictionary
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return config_dict

from termcolor import colored

def process_mbrl(args, env, cfg_train, logdir):
    from algorithms.td_mpc2.tdmpc2.tdmpc2.trainer.online_trainer import OnlineTrainer
    from algorithms.td_mpc2.tdmpc2.tdmpc2.common.parser import parse_cfg
    from algorithms.td_mpc2.tdmpc2.tdmpc2.common.seed import set_seed
    from algorithms.td_mpc2.tdmpc2.tdmpc2.common.buffer import Buffer
    from algorithms.td_mpc2.tdmpc2.tdmpc2.common.logger import Logger
    from algorithms.td_mpc2.tdmpc2.tdmpc2.tdmpc2 import TDMPC2
    from algorithms.td_mpc2.tdmpc2.tdmpc2.envs import make_env


    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.

    logdir = logdir + "_seed{}".format(env.task.cfg["seed"])

    torch.backends.cudnn.benchmark = True
    
    cfg = load_config(config_name='config')

    assert torch.cuda.is_available()
    assert cfg.steps > 0, 'Must train for at least 1 step.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

    trainer_cls = OnlineTrainer
    trainer = trainer_cls(
        cfg=cfg,
        env=make_env(cfg),
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )

    return trainer
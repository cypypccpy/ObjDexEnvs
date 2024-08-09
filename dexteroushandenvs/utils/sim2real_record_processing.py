# Third Party
import torch

# cuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    )
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

tensor_args = TensorDeviceType()
world_file = "collision_cage.yml"

robot_file = "ur10e.yml"
robot_cfg = RobotConfig.from_dict(
    load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
)

initial_joint_angles = [-0.4235312584306925, -1.8417856713793022, 2.1118022259904565, -0.26705746630618066, 1.1434836562123438, -3.150733285519455]  # 举例：设置机器人的初始关节角度
robot_cfg.initial_joint_positions = initial_joint_angles

world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
ik_config = IKSolverConfig.load_from_robot_config(
    robot_cfg,
    world_cfg,
    rotation_threshold=0.05,
    position_threshold=0.005,
    num_seeds=22,
    self_collision_check=True,
    self_collision_opt=True,
    tensor_args=tensor_args,
    use_cuda_graph=True,
)
ik_solver = IKSolver(ik_config)

q_sample = ik_solver.sample_configs(1)
# kin_state = ik_solver.fk(q_sample)
q_sample = torch.tensor([[3.4991441036750577, -1.310780687961321, -2.128748927522598, -2.84180679300243, -1.2157104341775433, 3.1342631916289605]],
       device='cuda:0')
kin_state = ik_solver.fk(q_sample)
print(kin_state.ee_position)
print(kin_state.ee_quaternion)
exit()
goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
result = ik_solver.solve_batch(goal)
q_solution = result.solution[result.success]

print(q_solution)
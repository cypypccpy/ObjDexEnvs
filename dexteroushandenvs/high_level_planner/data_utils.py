from matplotlib.pyplot import axis
import numpy as np
import os
import random
import torch
import pickle
from utils.torch_jit_utils import *

from torch import nn
from scipy.interpolate import interp1d
import open3d as o3d

def quaternion_to_rotation_matrix_for_f(q):
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
    R_pred = quaternion_to_rotation_matrix_for_f(q_pred.to(device))
    R_gt = quaternion_to_rotation_matrix_for_f(q_gt.to(device))

    # 计算旋转矩阵的差
    diff = R_pred - R_gt

    # 计算差矩阵的 Frobenius 范数
    frobenius_norm = torch.norm(diff, p='fro')

    return frobenius_norm.item()

@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

class DataLoader():
    def __init__(self, gym, sim, used_seq_list, functional, used_sub_seq_list, object_name, device, interpolate_time=1) -> None:
        self.seq_list = []
        self.texture_list = []
        self.used_seq_list = used_seq_list
        self.used_sub_seq_list = used_sub_seq_list
        self.functional = functional
        self.object_name = object_name
        self.device = device
        self.gym = gym
        self.sim = sim
        
        self.interpolate_time = interpolate_time
    
    def load_arctic_data(self, arctic_processed_path, arctic_raw_data_path):
        home = os.path.expanduser('~')
        
        self.seq_list = []
        self.texture_list = []
        
        self.obj_name_seq = []
        
        num_total_traj = len(self.used_seq_list) * len(self.used_sub_seq_list) * len(self.object_name) * len(self.functional)
        traj_count = 0
        
        for functional in self.functional:
            for seq_i in self.used_seq_list:
                for j in self.used_sub_seq_list:
                    for object_name in self.object_name:
                        traj_count += 1
                        progress = int(traj_count / num_total_traj * 100)
                        print("\r", end="")
                        print(
                            "Process progress: {}%: ".format(progress),
                            "▋" * (progress // 2),
                            end="",
                        )
                        
                        mano_p = "{}/s{}/{}_{}_{}.mano.npy".format(
                            arctic_raw_data_path, seq_i, object_name, functional, j
                        )
                        obj_p = "{}/s{}/{}_{}_{}.object.npy".format(
                            arctic_raw_data_path, seq_i, object_name, functional, j
                        )
                        # MANO

                        try:
                            data = np.load(
                                mano_p,
                                allow_pickle=True,
                            ).item()
                        except:
                            continue
                        
                        self.obj_name_seq.append(object_name)

                        mano_processed = (
                            "{}/seqs/s{}/{}_{}_{}.npy".format(
                                arctic_processed_path, seq_i, object_name, functional, j
                            )
                        )
                        self.mano_processed_data = np.load(
                            mano_processed,
                            allow_pickle=True,
                        ).item()

                        num_frames = len(data["right"]["rot"])

                        view_idx = 1

                        table_texture_files = (
                            "../assets/arctic_assets/object_vtemplates/{}/material.jpg".format(
                                object_name
                            )
                        )

                        table_texture_handle = self.gym.create_texture_from_file(
                            self.sim, table_texture_files
                        )
                        
                        self.texture_list.append(table_texture_handle)

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
                            
                        self.rot_r = to_torch(rot_r, device=self.device)[self.begin_frame :]
                        self.trans_r = to_torch(trans_r, device=self.device)[self.begin_frame :]
                        self.rot_l = to_torch(rot_l, device=self.device)[self.begin_frame :]
                        self.trans_l = to_torch(trans_l, device=self.device)[self.begin_frame :]
                        self.obj_params = to_torch(obj_params, device=self.device)[self.begin_frame :]

                        self.obj_rot_quat = to_torch(obj_rot_quat, device=self.device)[self.begin_frame :]
                        self.rot_r_quat = to_torch(rot_r_quat, device=self.device)[self.begin_frame :]
                        self.rot_l_quat = to_torch(rot_l_quat, device=self.device)[self.begin_frame :]
                        
                        self.left_middle_finger = to_torch(left_middle_finger, device=self.device).contiguous().view(left_middle_finger.shape[0], 15)[self.begin_frame :]
                        self.right_middle_finger = to_torch(right_middle_finger, device=self.device).contiguous().view(right_middle_finger.shape[0], 15)[self.begin_frame :]

                        self.rot_r_tem = self.rot_r.clone()
                        self.trans_r_tem = self.trans_r.clone()
                        self.rot_l_tem = self.rot_l.clone()
                        self.trans_l_tem = self.trans_l.clone()
                        self.obj_params_tem = self.obj_params.clone()

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
                            
                        self.left_fingertip = to_torch(left_fingertip, device=self.device).view(left_fingertip.shape[0], 15)[self.begin_frame :]
                        self.right_fingertip = to_torch(right_fingertip, device=self.device).view(right_fingertip.shape[0], 15)[self.begin_frame :]
                            
                        # transform quat for arm
                        right_transform_quat = to_torch(
                            [0.0, -0.707, 0.0, 0.707], dtype=torch.float, device=self.device
                        ).repeat((self.rot_r_quat.shape[0], 1))
                        left_transform_quat = to_torch(
                            [0.707, 0.0, 0.707, 0.0], dtype=torch.float, device=self.device
                        ).repeat((self.rot_l_quat.shape[0], 1))
                        self.rot_l_quat = quat_mul(self.rot_l_quat, left_transform_quat)
                        self.rot_r_quat = quat_mul(self.rot_r_quat, right_transform_quat)                     
                        
                        interpolate_time = self.interpolate_time
                        
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
                        
                        # fine-tuning
                        for i, rot_quat in enumerate(self.obj_rot_quat):
                            if i > 0:
                                if calculate_frobenius_norm_of_rotation_difference(rot_quat, last_obj_rot_global, device=self.device) > 0.5:
                                    self.obj_rot_quat[i] = last_obj_rot_global.clone()
                                
                            last_obj_rot_global = rot_quat.clone()

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
                        
        return self.seq_list, self.texture_list, self.obj_name_seq
                    
def interpolate_tensor(input_tensor, interpolate_time):
    """
    """
    batch_size = input_tensor.size(0)
    
    original_shape = input_tensor.size()
    new_batch_size = (original_shape[0] - 1) * interpolate_time + original_shape[0]
    new_shape = (new_batch_size, original_shape[1])

    interpolated_data = np.zeros((new_shape[0], new_shape[1]))
    for i in range(new_shape[1]):
        x = np.linspace(0, batch_size - 1, batch_size)  
        y = input_tensor[:, i].cpu().numpy()
        f = interp1d(x, y, kind='linear')
        new_x = np.linspace(0, batch_size - 1, new_batch_size)
        interpolated_data[:, i] = f(new_x)
    
    output_tensor = torch.tensor(interpolated_data, dtype=input_tensor.dtype, device=input_tensor.device)

    return output_tensor
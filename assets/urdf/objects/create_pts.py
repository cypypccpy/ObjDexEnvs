import open3d as o3d
import numpy as np
import pickle

mesh = o3d.io.read_triangle_mesh("/home/user/DexterousHandEnvs/assets/urdf/objects/meshes/cube_multicolor.obj")

# Sample points from the mesh surface
pcd = mesh.sample_points_uniformly(number_of_points=200)

# Convert Open3D point cloud to numpy array
downsampled_points = np.asarray(pcd.points) * 0.05

# Save data as a .pkl file
with open('/home/user/DexterousHandEnvs/assets/urdf/objects/point_cloud_200_pts.pkl', 'wb') as f:
    pickle.dump(downsampled_points, f)

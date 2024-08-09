# Bi-DexHands: Bimanual Dexterous Manipulation via Reinforcement Learning

ShadowHandFreeWithPhysics:

```python
python train_rlgames.py --task ShadowHandFreeWithPhysics --algo arctic_grasp --seed 22 --num_envs=1 --play --enable_camera
```

ShadowHandFreeVisualization:

```python
python train_rlgames.py --task ShadowHandFreeVisualization --algo arctic_grasp --seed 22 --num_envs=1 --play --hand=mano --traj_index=01_01
```
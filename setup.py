"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym",
    "torch",
    "matplotlib==3.5.1",
    "tqdm",
    "ipdb",
    "dowel",
    "akro",
    "pyyaml",
    "einops",
    "opencv-python",
    "open3d",
    "pytorch3d",
    "rl-games==1.5.2",
    "pybullet"
]


# Installation operation
setup(
    name="dexteroushandenvs",
    author="",
    version="0.1.0",
    description="Benchmark environments for Dexterous Hand in NVIDIA IsaacGym.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7"],
    zip_safe=False,
)

# EOF

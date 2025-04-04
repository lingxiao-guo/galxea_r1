conda create -n galaxea_act python=3.8
conda activate galaxea_act
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchvision 
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco
pip install dm_control
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install h5py_cache
pip install gnupg
pip install zarr
pip install -e .
# for diffusion policy
pip install omegaconf
pip install --upgrade diffusers[torch]
pip install hydra-core --upgrade

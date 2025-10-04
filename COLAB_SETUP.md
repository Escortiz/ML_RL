Colab setup for ML_RL project

This file lists ready-to-run Colab commands (bash/python cells) that install dependencies and run the repo in a headless Colab environment.

1) Mount Drive (optional)
```python
from google.colab import drive
drive.mount('/content/drive')
```

2) Clone repository (or copy from Drive)
```bash
# clone
!git clone https://github.com/<your-username>/<repo>.git repo
%cd repo
# or, if using Drive, change directory to your Drive location
%cd /content/drive/MyDrive/path/to/ML_RL
```

3) System deps for rendering & builds
```bash
# update and install system libs
!apt-get update -y
!apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 swig ffmpeg
```

4) Upgrade pip/build tools and install PyTorch (choose CPU or GPU)
```bash
# upgrade packaging tools
!pip install -U pip setuptools wheel
!pip install -U cython

# install torch (CPU) - if you want GPU, use the URL from pytorch.org
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

5) Install Python requirements (Colab-adapted)
```bash
!pip install -r requirements_colab.txt
```

6) MuJoCo note
- This repository originally lists `mujoco-py`, which often fails to install in Colab.
- If your env uses MuJoCo and supports the modern `mujoco` package, install it manually:
```bash
!pip install mujoco
# and set environment variable for headless rendering
%env MUJOCO_GL=egl
```
- If your environment uses `mujoco-py`, you will likely need to install additional system dependencies and a license key which is more complex.

7) Smoke test (in Python cell)
```python
import torch
from agent import Policy
p = Policy(11, 3)
s = torch.zeros(1, 11)
d, v = p(s)
print('mean', d.mean.shape, 'std', d.stddev.shape, 'value', v.shape)
```

8) Run training/test
```bash
# run a short test/train
!python test.py --episodes 2 --model ./model.mdl
# or train
!python train.py --n-episodes 10 --device cpu
```

9) Video recording
- If you use RecordVideo, output files will be in the repo folder. Move them to Drive if needed:
```bash
!cp -r videos_test_agent /content/drive/MyDrive/some_folder/
```

Troubleshooting
- If `pip install -r requirements_colab.txt` fails on `gym`, try installing a different version: `pip install gym==0.26.2`.
- If `mujoco` is required and fails, consult mujoco installation docs; consider running locally where MuJoCo is already set up.

Contact
- If any install step errors, paste the full pip error output here and I'll help debug.

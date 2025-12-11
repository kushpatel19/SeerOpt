<p align="center">
  <h1 align="center">SeerOpt: Bidirectional Learning and Optimization for Future-Aware Driving</h1>

  <p align="center">
    <a href="https://www.linkedin.com/in/harsh-pandit-6a8a081b5/"><strong>Harsh Pandit</strong></a> ·
    <a href="https://www.linkedin.com/in/kush-patel-5397281b8/"><strong>Kush Patel</strong></a> ·
    <a href="https://www.linkedin.com/in/kunal-siddhawar-858839140/"><strong>Kunal Siddhawar</strong></a>
  </p>
  
  <h3 align="center">
    <a href="media/SeerOpt.pdf">Paper</a> 
    <!-- <a href="https://www.youtube.com/watch?v=dF_nQ6IA1po">YouTube</a> -->
  
  </h3>

  <p align="center">
    <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.9+-blue" />
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0-red" />
    <img alt="KITTI" src="https://img.shields.io/badge/Dataset-NAVSIM-green" />
    <img alt="GTSAM" src="https://img.shields.io/badge/Optimization-GTSAM-orange" />
  </p>

  <p align="center">
    <img src="./media/Poster.jpeg" alt="SeerOpt Poster" width="850"/>
  </p>
</p>

---

### 📌 Project Summary

**SeerOpt** is a future-aware planning framework that enhances the safety and feasibility of neural autonomous driving models. Building on the [SeerDrive](https://github.com/LogosRoboticsGroup/SeerDrive/tree/main?tab=readme-ov-file) baseline , which jointly predicts future BEV scenes and trajectories, SeerOpt introduces a robust post-processing layer using [GTSAM](https://gtsam.org/) factor graph optimization and Nonlinear Model Predictive Control (NMPC). By enforcing strict kinematic constraints and leveraging extracted drivable area maps for obstacle avoidance, SeerOpt bridges the gap between flexible neural predictions and reliable control. Evaluation on the [NAVSIM](https://github.com/autonomousvision/navsim) benchmark demonstrates that our optimization-based refinement reduces collision rates thus improves overall PDM score compared to pure learning-based methods.

---

## 🚀 Getting Started

### ⚙️ Install the Environment
```bash
git clone https://github.com/kushpatel19/SeerOpt.git
cd SeerDrive/
conda env create -f environment.yml
conda activate seeropt

git clone git@github.com:motional/nuplan-devkit.git
cd nuplan-devkit/
pip install -e .

pip install -e .

pip install diffusers einops 
pip install rich==14.0.0
```

### 🔍 Prepare the data
```bash
cd navsim
cd download && ./download_maps
./download_test
./download_navhard_two_stage
```

### ⬇️ Download Checkpoints

Download the pre-trained model weights from the link below:
1. [**SeerDrive Checkpoints**](https://drive.google.com/file/d/1CvFsVnMhJCHZ21rTFcOKkgHHrJjteXLb/view?usp=sharing)
2. [**ResNet34 Weights**](https://drive.google.com/drive/folders/1dIHK8nXkzhIhGCRQOpKibaizwH-7fHqs)

Once downloaded, create a `ckpt` directory inside the `SeerDrive` folder and move both file into it.

```bash
cd SeerDrive
mkdir ckpt
# Move the downloaded file into the new directory (adjust the source path as needed)
mv ~/Downloads/SeerDrive_checkpoints.ckpt ckpt/
```

### 📂 Organize Data

Once the dataset download is complete, create a `data` folder inside the `SeerDrive` directory. Move and organize your downloaded files to match the structure below. This ensures the paths in the config files resolve correctly.

```angular2html
SeerDrive
├── ckpt
│   └── SeerDrive_checkpoints.ckpt
│   └── resnet34.pth
├── data
│   └── navsim
│   │   ├── navhard_two_stage
│   │   ├── navsim_logs
│   │   ├── sensor_blobs
│   └── nuplan_maps
│       └── maps
│           ├── .maplocks
│           ├── sg-one-north
│           ├── us-ma-boston
│           ├── us-nv-las-vegas-strip
│           ├── us-pa-pittsburgh-hazelwood
│           └── nuplan-maps-v1.0.json
```

```bash
cd SeerDrive
python scripts/misc/k_means_trajs.py
bash scripts/evaluation/run_metric_caching.sh
# testing
bash scripts/training/seerdrive_eval.sh
```



### 🔧 How to Run
```bash


```

### 🔧 Install Other Dependancy
```bash
# Install conda packages
conda install pillow=11.0.0 pandas=2.2.3 opencv=4.11.0 scipy=1.13.1 numpy=1.26.4 -c conda-forge

# Install GTSAM
conda install -c conda-forge gtsam=4.2.0

# Install HuggingFace Transformers
pip install transformers==4.50.3
```

### 📚 KITTI Dataset
We evaluate our pipeline on the KITTI Odometry Benchmark, leveraging both the raw image sequences and the official ground‑truth trajectories. To set up:

1. Download the **Grayscale Odometry Sequences** (≈22 GB) and the corresponding **ground‑truth poses** (≈4 MB) from the [KITTI Odometry Benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).  
2. In `main.py`, set:
   ```python
   groundtruth_file = "/path/to/poses.txt"
   image_folder     = "/path/to/image/sequences/"
3. Ensure that your folder structure matches KITTI’s format so that each frame aligns correctly with its ground‑truth pose.

### 📚 Pre-trained Model
We build on MagicLeap’s **SuperGluePretrainedNetwork** for feature matching:

1. Clone the official repo into your working directory:
```bash
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
```
2. Verify that the `SuperGluePretrainedNetwork` folder sits alongside `main.py` so imports resolve cleanly.
3. The supplied weights (for both SuperPoint and SuperGlue) will be automatically downloaded on first run—no additional steps required.


### How to Run
```bash
python main.py
```
### Licence

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full terms.  
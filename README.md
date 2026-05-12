# SAGE: Plan in Sandbox, Navigate in Open Worlds

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue)](https://icml.cc/Conferences/2026)
[![arXiv](https://img.shields.io/badge/arXiv-2605.10118-b31b1b)](https://arxiv.org/abs/2605.10118)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official source code for the ICML 2026 paper:

**Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation**

SAGE (Sandbox-Abstracted Grounded Experience) learns embodied navigation behavior from physics-grounded semantic abstractions instead of relying only on photorealistic simulation. The codebase follows the paper's three-stage pipeline: **Genesis** for sandbox task and experience synthesis, **Evolution** for reinforcement-learning-based policy distillation, and **Navigation** for open-world embodied evaluation.

> Before an embodied agent steps into the open world, SAGE lets it rehearse: not in a perfect digital twin, but in a compact sandbox where physics, semantics, and experience can be distilled into useful navigation priors.

## Preparations

#### Dataset

Please download the train and val split of [HM3D](https://aihabitat.org/datasets/hm3d-semantics/), and specify
the path in `cfg/eval_aeqa.yaml` and `cfg/eval_goatbench.yaml`. For example, if your download path is `/your_path/hm3d/` that 
contains `/your_path/hm3d/train/` and `/your_path/hm3d/val/`, you can set the `scene_data_path` in the config files as `/your_path/hm3d/`.
For GOAT-Bench, we include the complete `val_unseen` split in this repository.

Then download [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS) to `data/` folder.



## *Genesis*

### 1 - Setup

Set up the conda environment (Linux, Python 3.9):

```bash
conda create -n sagenav python=3.9 -y && conda activate sagenav

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge -c aihabitat habitat-sim=0.2.5 headless faiss-cpu=1.7.4 -y
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2 -y

pip install omegaconf==2.3.0 open-clip-torch==2.26.1 ultralytics==8.2.31 supervision==0.21.0 opencv-python-headless==4.10.* \
 scikit-learn==1.4 scikit-image==0.22 open3d==0.18.0 hipart==1.0.4 openai==1.35.3 httpx==0.27.2                                                      

```

### 2 - Run Sandbox Task and Experience Synthesis 

Run the following script to synthesize the HM3D sandbox task and experience (fullset in our paper):

```
python process_HM3D_data.py
```

Run the following scripts to synthesize the InteriorGS sandbox task and experience (fullset in our paper):

```
python batch_process_interiorgs.py
python process_InteriorGS_data.py
```



## *Evolution*

### 1 - Setup

Set up a new conda environment (Linux, Python 3.10):

```
conda create -n sage python=3.10 -y && conda activate sage
pip install -e .
```

### 2 - Run Training

Run the following scripts to train a 2B *SAGE* agent:

```
cd Evolution
bash examples\qwen3_vl_2b_nav_v3_grpo_mixed.sh
```



## *Navigation*

### 1 - Setup

Please set up the endpoint and API key for the vLLM server in `src/const.py`.

### 2 - Run Evaluation on A-EQA

First run the following script to generate the predictions for the A-EQA dataset:

```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml
```
To split tasks, you can add `--start_ratio` and `--end_ratio` to specify the range of tasks to evaluate. For example,
to evaluate the first half of the dataset, you can run:
```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml --start_ratio 0.0 --end_ratio 0.5
```
After the scripts finish, the results from all splits will be automatically aggregated and saved.

### 3 - Run Evaluation on GOAT-Bench
You can directly run the following script:
```bash
python run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml
```
The results will be saved and printed after the script finishes. You can also split the task similarly by adding `--start_ratio` and `--end_ratio`.
Note that GOAT-Bench provides 10 explore episodes for each scene, and by default we only test the first episode due to the time and resource constraints.
You can also specify the episode to evaluate for each scene by setting `--split`.



## Citation

If you find *SAGE* useful for embodied navigation, sandbox experience synthesis, or open-world planning research, please cite our ICML 2026 paper:

```bibtex
@inproceedings{shen2026sage,
  title     = {Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation},
  author    = {Shen, Zhixuan and Du, Jiawei and Guo, Ziyu and Luo, Han and Peng, Lilan and Zhou, Joey Tianyi and Luo, Haonan and Li, Tianrui},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  note      = {ICML 2026}
}
```

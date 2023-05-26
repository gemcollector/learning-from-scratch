# On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline
This is the repo of the paper: [On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline](https://arxiv.org/abs/2212.05749)



## Setup
Our code is based on the [isaacgym](https://developer.nvidia.com/isaac-gym) and the repo of [Masked Visual Pre-training for Motor Control](https://github.com/ir413/mvp).
- Create a conda environment:
```
conda create -n mvp python=3.7
conda activate mvp
```
- Install Pytorch:
    - `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`
- Install the Preview 3 isaac gym version:
  - download issacgym from https://developer.nvidia.com/isaac-gym/download, and follow the instructions to install it.
  - For anaconda users, there  are some installing tips in `isaacgym/docs/install.html` for handling potential issues (e.g., `libpython3.7`).
  
- Install MVP:
    - `pip install -r requirements.txt`
    - `pip install -e .`

  


## LfS
If you want reproduce the results of LfS, just run the following commands:
```
bash script/train_w_DA.sh
```
The setting of LfS(w/o aug) is:
` is_image=True is_DA=False is_drac=False`

The setting of LfS(+ aug) is:
` is_image=True is_DA=True is_drac=True`

## Pre-trained baseline
The pretrained baselines contains MVP, PVR (Moco) and R3M. 

For R3M, we can just follow the setup from the official link:
`https://github.com/facebookresearch/r3m`

For PVR, we use the `moco_aug.pth.tar` from the official link:
`https://github.com/sparisi/pvr_habitat/releases/download/models/moco_aug.pth.tar`

For MVP, we choose `vits-mae-hoi` as the pretraiend model, and it can be downloaded from:
```
https://github.com/ir413/mvp
```

Besdies, you should change the line of 135 in `actor_critic.py` to your own model path.


## Off-policy 
The off-policy part is mainly based on the code of [DrQ-v2](). Our modifications have been indicated in the appendix, and implementation can be done very easily.


## Citation
<a name="citation"></a>
If you find our work useful in your research, please consider citing our work as follows:

```
@article{hansen2022pre,
  title={On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline},
  author={Hansen, Nicklas and Yuan, Zhecheng and Ze, Yanjie and Mu, Tongzhou and Rajeswaran, Aravind and Su, Hao and Xu, Huazhe and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2212.05749},
  year={2022}
}
```
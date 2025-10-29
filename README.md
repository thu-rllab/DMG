# Doubly Mild Generalization for Offline Reinforcement Learning

Code for NeurIPS 2024 accepted paper: [**Doubly Mild Generalization for Offline Reinforcement Learning**](https://arxiv.org/pdf/2411.07934).

## 🔧 Environment
Paper results were collected with [MuJoCo 210](https://mujoco.org/) (and [mujoco-py 2.1.2.14](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.23.1](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/Farama-Foundation/D4RL). Networks are trained using [PyTorch 1.11.0](https://github.com/pytorch/pytorch) and [Python 3.7](https://www.python.org/).

## 🚀 Usage


### Offline RL Training

Use the following command to train offline RL on D4RL, including Gym locomotion and Antmaze tasks, and save the models.
```
python train_offline.py --env halfcheetah-medium-v2 --lam 0.25 --nu 0.1 --save_model
python train_offline.py --env antmaze-large-diverse-v2 --lam 0.25 --nu 0.5 --no_normalize --save_model
```

### Offline-to-Online Finetuning

Use the following command to online fine-tune the pretrained offline models on AntMaze tasks.
```
python train_finetune.py --env antmaze-large-diverse-v2 --lam 0.25 --nu 0.5 --lam_end 0.5 --nu_end 0.005 --no_normalize
```

### Logging

You can view saved runs using TensorBoard.

```
tensorboard --logdir <run_dir>
```

## 📝 Citation

If you find this work useful, please consider citing:
```bibtex
@article{mao2024doubly,
  title={Doubly mild generalization for offline reinforcement learning},
  author={Mao, Yixiu and Wang, Qi and Qu, Yun and Jiang, Yuhang and Ji, Xiangyang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={51436--51473},
  year={2024}
}
```
# Doubly Mild Generalization for Offline Reinforcement Learning

Implementation of the DMG algorithm.

## Environment
Paper results were collected with [MuJoCo 210](https://mujoco.org/) (and [mujoco-py 2.1.2.14](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.23.1](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/Farama-Foundation/D4RL). Networks are trained using [PyTorch 1.11.0](https://github.com/pytorch/pytorch) and [Python 3.7](https://www.python.org/).

## Usage


### Offline RL Training

Use the following command to train offline RL on D4RL, including Gym locomotion and Antmaze tasks, and save the models.
```
python train_offline.py --env halfcheetah-medium-v2 --lam 0.25 --nu 0.1 --save_model
python train_offline.py --env antmaze-large-play-v2 --lam 0.25 --nu 0.5 --no_normalize --save_model
```

### Offline-to-Online Finetuning

Use the following command to online fine-tune the pretrained offline models on AntMaze tasks.
```
python train_finetune.py --env antmaze-large-diverse-v2 --lam 0.25 --nu 0.5 --no_normalize
```

### Logging

You can view saved runs using TensorBoard.

```
tensorboard --logdir <run_dir>
```
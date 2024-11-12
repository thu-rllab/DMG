import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import random
import json
import utils
import DMG
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from tqdm import trange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def snapshot_src(src, target, exclude_from):
    try:
        os.mkdir(target)
    except OSError:
        pass
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")
    
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)
	eval_env.action_space.seed(seed + seed_offset)
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="hopper-medium-v2")        # Environment name
	parser.add_argument("--seed", default=3, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=None, type=int)      # Frequency of Evaluation
	parser.add_argument("--eval_episodes", default=None, type=int)  # Evaluation episodes
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--no_normalize", action="store_true")      # Not normalize states
	parser.add_argument("--lam", default=0.25, type=float)          # DMG parameter /lambda used in offline RL
	parser.add_argument("--nu", default=0.5, type=float)            # DMG parameter /nu used in offline RL
	parser.add_argument("--save_model", action="store_true")        # Save trained models
	args = parser.parse_args()

	print("---------------------------------------")
	print(f"Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	env = gym.make(args.env)
	work_dir = './runs/offline/{}/lam{}_nu{}_seed{}'.format(
     args.env, args.lam, args.nu, args.seed)
	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	writer = SummaryWriter(work_dir)
	with open(os.path.join(work_dir, 'args.json'), 'w') as f:
		json.dump(vars(args), f, sort_keys=True, indent=4)
	snapshot_src('.', os.path.join(work_dir, 'src'), '.gitignore')

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

	if not args.no_normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	if 'antmaze' in args.env:
		replay_buffer.reward = np.where(replay_buffer.reward == 1.0, 0.0, -1.0)
		antmaze = True
		args.eval_episodes = 100 if args.eval_episodes is None else args.eval_episodes
		args.eval_freq = 50000 if args.eval_freq is None else args.eval_freq
		expectile = 0.9 # follow IQL
		temp = 10.0 # follow IQL
	else:
		antmaze = False
		args.eval_episodes = 10 if args.eval_episodes is None else args.eval_episodes
		args.eval_freq = 20000 if args.eval_freq is None else args.eval_freq
		expectile = 0.7 # follow IQL
		temp = 3.0 # follow IQL

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"replay_buffer": replay_buffer,
		"antmaze": antmaze,
		"discount": args.discount,
		"tau": args.tau,
		"policy_freq": args.policy_freq,
		"expectile": expectile,
		"temp": temp,
		"lam": args.lam,
		"nu": args.nu,
	}

	policy = DMG.DMG(**kwargs)
	
	for t in trange(int(args.max_timesteps)):
		policy.train_offline(args.batch_size, writer)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			d4rl_score = eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=args.eval_episodes)
			writer.add_scalar('eval/d4rl_score', d4rl_score, t)
	if args.save_model:
		policy.save(work_dir)
	time.sleep( 10 )

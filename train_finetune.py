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
	parser.add_argument("--env", default="antmaze-umaze-v2")        # Environment name
	parser.add_argument("--seed", default=5, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=None, type=int)      # Frequency of Evaluation
	parser.add_argument("--eval_episodes", default=None, type=int)  # Evaluation episodes
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--no_normalize", action="store_true")      # Not normalize states
	parser.add_argument("--lam", default=0.25, type=float)          # DMG parameter /lambda used in offline RL
	parser.add_argument("--nu", default=0.5, type=float)            # DMG parameter /nu used in offline RL
	parser.add_argument("--save_model", action="store_true")        # Save trained models

	# Offline-to-online Finetune
	parser.add_argument("--lam_end", default=0.5, type=float)       # Final value of /lambda after decay
	parser.add_argument("--nu_end", default=0.005, type=float)      # Final value of /nu after decay
	parser.add_argument('--buffer_size', default=2000000, type=int) # Maximum buffer size for offline and online transitions
	parser.add_argument('--start_timesteps', default=0, type=int)   # Online interaction steps before starting RL training
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian noise used in online exploration
	args = parser.parse_args()

	print("---------------------------------------")
	print(f"Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	env = gym.make(args.env)
	model_dir = './runs/offline/{}/lam{}_nu{}_seed{}'.format(
     args.env, args.lam, args.nu, args.seed)
	work_dir = './runs/finetune/{}/lam{}_nu{}_lamend{}_nuend{}_seed{}'.format(
     args.env, args.lam, args.nu, args.lam_end, args.nu_end, args.seed)
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

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)
	replay_buffer.convert_D4RL_finetune(d4rl.qlearning_dataset(env))
	assert replay_buffer.size + args.max_timesteps <= replay_buffer.max_size

	if args.no_normalize:
		mean,std = 0,1
	else:
		assert False
        # TODO: normalize to self.state[:self.ptr]
		mean,std = replay_buffer.normalize_states()

	if 'antmaze' in args.env:
		replay_buffer.reward = np.where(replay_buffer.reward == 1.0, 0.0, -1.0) # follow D4RL instructions
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

	discount = 0.995 if 'antmaze-large' in args.env else 0.99 # follow several previous works

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"replay_buffer": replay_buffer,
		"discount": discount,
		"tau": args.tau,
		"policy_freq": args.policy_freq,
		"expectile": expectile,
		"temp": temp,
		"lam": args.lam,
		"lam_end": args.lam_end,
		"nu": args.nu,
		"nu_end": args.nu_end,
	}

	policy = DMG.DMG(**kwargs)

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	episode_state = [state]

	# Load offline pretrained model
	policy.load(model_dir)
	
	for t in trange(int(args.max_timesteps)):
		episode_timesteps += 1

		action = (
			policy.select_action(state) + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
		).clip(-max_action, max_action)

		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		if 'antmaze' in args.env:
			reward_original = reward
			reward = 0.0 if done_bool else -1.0

		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward_original
		episode_state.append(state)

		if t >= args.start_timesteps:
			policy.train_online(args.batch_size, writer)

		if done:
			print(
				f"Time steps: {t+1} Episode Num: {episode_num+1} Episode Timesteps: {episode_timesteps} Episode Return: {episode_reward:.3f} Last Reward: {reward_original:.3f}")
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			episode_state = [state]

		# Evaluate episode
		if t == 0 or (t + 1) % args.eval_freq == 0:
			print(f"Time steps: {t+1}")
			d4rl_score = eval_policy(policy, args.env, args.seed, mean, std, eval_episodes=args.eval_episodes)
			writer.add_scalar('eval/d4rl_score', d4rl_score, t)
	if args.save_model:
		policy.save(work_dir)
	time.sleep( 10 )

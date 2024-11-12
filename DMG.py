import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def expectile_loss(diff, expectile=0.7):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], -1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

class ValueCritic(nn.Module):
	def __init__(self, state_dim):
		super(ValueCritic, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state):
		sa = torch.cat([state], -1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class DMG(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		replay_buffer,
		discount=0.99,
		tau=0.005,
		policy_freq=2,
		antmaze=True,
		expectile=0.9,
		temp = 10.0,
		lam=0.25,
		lam_end=0.5,
		nu = 0.5,
		nu_end = 0.005,
	):
		
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.critic_target = copy.deepcopy(self.critic)
		self.value_critic = ValueCritic(state_dim).to(device)
		self.value_critic_optimizer = torch.optim.Adam(self.value_critic.parameters(), lr=3e-4)
        
		self.replay_buffer = replay_buffer
		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.policy_freq = policy_freq
		self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, int(int(1e6)/self.policy_freq))
		self.lam = lam
		self.lam_end = lam_end
		self.expectile = expectile
		self.temp = temp
		self.nu = nu
		self.nu_end = nu_end
		self.max_weight = 100 if antmaze else 3
		self.total_it = 0
		self.decay_rate = 1
		self.exp_decay = 0.99


	def select_action(self, state):
		with torch.no_grad():
			self.actor.eval()
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			action = self.actor(state).cpu().data.numpy().flatten()
			self.actor.train()
			return action

	def train_offline(self, batch_size=256, writer=None):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

		# value_critic
		with torch.no_grad():
			iql_q1, iql_q2 = self.critic_target(state, action)
			iql_q = torch.cat([iql_q1, iql_q2],dim=1)
			iql_q,_ = torch.min(iql_q,dim=1,keepdim=True)
		iql_v = self.value_critic(state)
		value_loss = expectile_loss(iql_q - iql_v, self.expectile).mean()
		self.value_critic_optimizer.zero_grad()
		value_loss.backward()
		self.value_critic_optimizer.step()

		# critic
		with torch.no_grad():
			noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
		# Compute the target Q value
		with torch.no_grad():
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q_pi = torch.cat([target_Q1, target_Q2],dim=1)
			target_Q_pi,_ = torch.min(target_Q_pi,dim=1,keepdim=True)
			target_Q_iql = self.value_critic(next_state)
			target_Q = reward + not_done * self.discount * (self.lam * target_Q_pi + (1-self.lam) * target_Q_iql)

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		critic_loss =  F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		if self.total_it % 10000 == 0:
			with torch.no_grad():
				writer.add_scalar('train/critic_loss', critic_loss.item(), self.total_it)
				curr_Q = torch.cat([current_Q1, current_Q2],dim=1)
				writer.add_scalar('train/Q', curr_Q.mean().item(), self.total_it)
				writer.add_scalar('train/iqlV', iql_v.mean().item(), self.total_it)
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			pi = self.actor(state)
			with torch.no_grad():
				awr_v = self.value_critic(state)
				awr_q1, awr_q2 = self.critic_target(state, action)
				awr_q = torch.minimum(awr_q1, awr_q2)
				exp_a = torch.exp((awr_q - awr_v) * self.temp)
				exp_a = torch.clamp(exp_a, max=self.max_weight).detach()

			v1,v2 = self.critic(state, pi)
			v = torch.cat([v1,v2], dim=1)
			vmin,_ = torch.min(v, dim=1)
			lmbda = 1.0 / vmin.abs().mean().detach() # follow TD3BC
			q_loss = -lmbda * vmin.mean()

			awr_loss = (exp_a * ((pi - action)**2)).mean()
			actor_loss = q_loss + self.nu * awr_loss

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			self.actor_lr_schedule.step()

			if self.total_it % 10000 == 0:
				writer.add_scalar('train/actor_loss', actor_loss.item(), self.total_it)

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def train_online(self, batch_size=256, writer=None):
		self.total_it += 1
		if self.total_it % 1000 == 0:
			self.decay_rate = self.decay_rate * self.exp_decay
		lam = self.lam_end - (self.lam_end - self.lam) * self.decay_rate
		nu = self.nu_end +  (self.nu - self.nu_end) * self.decay_rate

		# Sample replay buffer 
		state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

		# value_critic
		with torch.no_grad():
			iql_q1, iql_q2 = self.critic_target(state, action)
			iql_q = torch.cat([iql_q1, iql_q2],dim=1)
			iql_q,_ = torch.min(iql_q,dim=1,keepdim=True)
		iql_v = self.value_critic(state)
		value_loss = expectile_loss(iql_q - iql_v, self.expectile).mean()
		self.value_critic_optimizer.zero_grad()
		value_loss.backward()
		self.value_critic_optimizer.step()

		# critic
		with torch.no_grad():
			noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
		# Compute the target Q value
		with torch.no_grad():
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q_pi = torch.cat([target_Q1, target_Q2],dim=1)
			target_Q_pi,_ = torch.min(target_Q_pi,dim=1,keepdim=True)
			target_Q_iql = self.value_critic(next_state)
			target_Q = reward + not_done * self.discount * (lam * target_Q_pi + (1-lam) * target_Q_iql)

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		critic_loss =  F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		if self.total_it % 10000 == 0:
			with torch.no_grad():
				writer.add_scalar('train/critic_loss', critic_loss.item(), self.total_it)
				curr_Q = torch.cat([current_Q1, current_Q2],dim=1)
				writer.add_scalar('train/Q', curr_Q.mean().item(), self.total_it)
				writer.add_scalar('train/iqlV', iql_v.mean().item(), self.total_it)
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			pi = self.actor(state)
			with torch.no_grad():
				awr_v = self.value_critic(state)
				awr_q1, awr_q2 = self.critic_target(state, action)
				awr_q = torch.minimum(awr_q1, awr_q2)
				exp_a = torch.exp((awr_q - awr_v) * self.temp)
				exp_a = torch.clamp(exp_a, max=self.max_weight).detach()

			v1,v2 = self.critic(state, pi)
			v = torch.cat([v1,v2], dim=1)
			vmin,_ = torch.min(v, dim=1)
			lmbda = 1.0 / vmin.abs().mean().detach() # follow TD3BC
			q_loss = -lmbda * vmin.mean()

			awr_loss = (exp_a * ((pi - action)**2)).mean()
			actor_loss = q_loss + nu * awr_loss

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			if self.total_it % 10000 == 0:
				writer.add_scalar('train/actor_loss', actor_loss.item(), self.total_it)
				writer.add_scalar('train/lam', lam, self.total_it)
				writer.add_scalar('train/nu', nu, self.total_it)

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
	def save(self, model_dir):
		torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_s{str(self.total_it)}.pth"))
		torch.save(self.critic_target.state_dict(), os.path.join(model_dir, f"critic_target_s{str(self.total_it)}.pth"))
		torch.save(self.critic_optimizer.state_dict(), os.path.join(
			model_dir, f"critic_optimizer_s{str(self.total_it)}.pth"))

		torch.save(self.value_critic.state_dict(), os.path.join(model_dir, f"value_critic_s{str(self.total_it)}.pth"))
		torch.save(self.value_critic_optimizer.state_dict(), os.path.join(
			model_dir, f"value_critic_optimizer_s{str(self.total_it)}.pth"))

		torch.save(self.actor.state_dict(), os.path.join(model_dir, f"actor_s{str(self.total_it)}.pth"))
		torch.save(self.actor_target.state_dict(), os.path.join(model_dir, f"actor_target_s{str(self.total_it)}.pth"))
		torch.save(self.actor_optimizer.state_dict(), os.path.join(
			model_dir, f"actor_optimizer_s{str(self.total_it)}.pth"))

	def load(self, model_dir, step=1000000):
		self.critic.load_state_dict(torch.load(os.path.join(model_dir, f"critic_s{str(step)}.pth")))
		self.critic_target.load_state_dict(torch.load(os.path.join(model_dir, f"critic_target_s{str(step)}.pth")))
		self.critic_optimizer.load_state_dict(torch.load(os.path.join(model_dir, f"critic_optimizer_s{str(step)}.pth")))

		self.value_critic.load_state_dict(torch.load(os.path.join(model_dir, f"value_critic_s{str(step)}.pth")))
		self.value_critic_optimizer.load_state_dict(torch.load(os.path.join(model_dir, f"value_critic_optimizer_s{str(step)}.pth")))

		self.actor.load_state_dict(torch.load(os.path.join(model_dir, f"actor_s{str(step)}.pth")))
		self.actor_target.load_state_dict(torch.load(os.path.join(model_dir, f"actor_target_s{str(step)}.pth")))
		self.actor_optimizer.load_state_dict(torch.load(os.path.join(model_dir, f"actor_optimizer_s{str(step)}.pth")))
		self.actor_optimizer.param_groups[0]['lr'] = 3e-4
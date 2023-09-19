import math
import random
import time
import os
import sys
from datetime import datetime

import collections
from collections import namedtuple
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from pymss import EnvManager
from IPython import embed
from Model import *


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
Episode = namedtuple('Episode',('s','a','r', 'value', 'logprob'))
class EpisodeBuffer(object):
	def __init__(self):
		self.data = []

	def Push(self, *args):
		self.data.append(Episode(*args))
	def Pop(self):
		self.data.pop()
	def GetData(self):
		return self.data
MuscleTransition = namedtuple('MuscleTransition',('JtA','tau_des','L','b'))
class MuscleBuffer(object):
	def __init__(self, buff_size = 10000):
		super(MuscleBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(MuscleTransition(*args))

	def Clear(self):
		self.buffer.clear()
Transition = namedtuple('Transition',('s','a', 'logprob', 'TD', 'GAE'))
class ReplayBuffer(object):
	def __init__(self, buff_size = 10000):
		super(ReplayBuffer, self).__init__()
		self.buffer = deque(maxlen=buff_size)

	def Push(self,*args):
		self.buffer.append(Transition(*args))

	def Clear(self):
		self.buffer.clear()
class PPO(object):
	def __init__(self,meta_file):
		np.random.seed(seed = int(time.time()))
		self.num_slaves = 16
		self.env = EnvManager(meta_file,self.num_slaves)
		self.use_muscle = self.env.UseMuscle()
		self.num_state = self.env.GetNumState()
		self.num_action = self.env.GetNumAction()
		self.num_muscles = self.env.GetNumMuscles()

		self.num_epochs = 10
		self.num_epochs_muscle = 3
		self.num_evaluation = 0
		self.num_tuple_so_far = 0
		self.num_episode = 0
		self.num_tuple = 0
		self.num_simulation_Hz = self.env.GetSimulationHz()
		self.num_control_Hz = self.env.GetControlHz()
		self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

		self.gamma = 0.99
		self.lb = 0.99

		self.buffer_size = 2048
		self.batch_size = 128
		self.muscle_batch_size = 128
		self.replay_buffer = ReplayBuffer(30000)
		self.muscle_buffer = MuscleBuffer(30000)

		self.model = SimulationNN(self.num_state,self.num_action)

		self.muscle_model = MuscleNN(self.env.GetNumTotalMuscleRelatedDofs(),self.num_action,self.num_muscles)
		if use_cuda:
			self.model.cuda()
			self.muscle_model.cuda()

		self.default_learning_rate = 1E-4
		self.default_clip_ratio = 0.2
		self.learning_rate = self.default_learning_rate
		self.clip_ratio = self.default_clip_ratio
		self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
		self.optimizer_muscle = optim.Adam(self.muscle_model.parameters(),lr=self.learning_rate)
		self.max_iteration = 50000

		self.w_entropy = -0.001

		self.loss_actor = 0.0
		self.loss_critic = 0.0
		self.loss_muscle = 0.0
		self.rewards = []
		self.sum_return = 0.0
		self.max_return = -1.0
		self.max_return_epoch = 1
		self.tic = time.time()

		self.episodes = [None]*self.num_slaves
		for j in range(self.num_slaves):
			self.episodes[j] = EpisodeBuffer()
		self.env.Resets(True)

	def SaveModel(self):
		self.model.save('../nn/current.pt')
		self.muscle_model.save('../nn/current_muscle.pt')
		
		if self.max_return_epoch == self.num_evaluation:
			self.model.save('../nn/max.pt')
			self.muscle_model.save('../nn/max_muscle.pt')
		if self.num_evaluation%100 == 0:
			self.model.save('../nn/'+str(self.num_evaluation//100)+'.pt')
			self.muscle_model.save('../nn/'+str(self.num_evaluation//100)+'_muscle.pt')

	def LoadModel(self,path):
		self.model.load('../nn/'+path+'.pt')
		self.muscle_model.load('../nn/'+path+'_muscle.pt')

	def ComputeTDandGAE(self):
		self.replay_buffer.Clear()
		self.muscle_buffer.Clear()
		self.sum_return = 0.0
		for epi in self.total_episodes:
			data = epi.GetData()
			size = len(data)
			if size == 0:
				continue
			states, actions, rewards, values, logprobs = zip(*data)

			values = np.concatenate((values, np.zeros(1)), axis=0)
			advantages = np.zeros(size)
			ad_t = 0

			epi_return = 0.0
			for i in reversed(range(len(data))):
				epi_return += rewards[i]
				delta = rewards[i] + values[i+1] * self.gamma - values[i]
				ad_t = delta + self.gamma * self.lb * ad_t
				advantages[i] = ad_t
			self.sum_return += epi_return
			TD = values[:size] + advantages
			
			for i in range(size):
				self.replay_buffer.Push(states[i], actions[i], logprobs[i], TD[i], advantages[i])
		self.num_episode = len(self.total_episodes)
		self.num_tuple = len(self.replay_buffer.buffer)
		print('SIM : {}'.format(self.num_tuple))
		self.num_tuple_so_far += self.num_tuple

		muscle_tuples = self.env.GetMuscleTuples()
		for i in range(len(muscle_tuples)):
			self.muscle_buffer.Push(muscle_tuples[i][0],muscle_tuples[i][1],muscle_tuples[i][2],muscle_tuples[i][3])
	def GenerateTransitions(self):
		self.total_episodes = []
		states = [None]*self.num_slaves
		actions = [None]*self.num_slaves
		rewards = [None]*self.num_slaves
		states_next = [None]*self.num_slaves
		states = self.env.GetStates()
		local_step = 0
		terminated = [False]*self.num_slaves
		counter = 0
		while True:
			counter += 1
			if counter%10 == 0:
				print('SIM : {}'.format(local_step),end='\r')
			a_dist,v = self.model(Tensor(states))
			actions = a_dist.sample().cpu().detach().numpy()
			# actions = a_dist.loc.cpu().detach().numpy()
			logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
			values = v.cpu().detach().numpy().reshape(-1)
			self.env.SetActions(actions)
			if self.use_muscle:
				mt = Tensor(self.env.GetMuscleTorques())
				for i in range(self.num_simulation_per_control//2):
					dt = Tensor(self.env.GetDesiredTorques())
					activations = self.muscle_model(mt,dt).cpu().detach().numpy()
					self.env.SetActivationLevels(activations)

					self.env.Steps(2)
			else:
				self.env.StepsAtOnce()

			for j in range(self.num_slaves):
				nan_occur = False
				terminated_state = True

				if np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or np.any(np.isnan(states[j])) or np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j])):
					nan_occur = True
				
				elif self.env.IsEndOfEpisode(j) is False:
					terminated_state = False
					rewards[j] = self.env.GetReward(j)
					self.episodes[j].Push(states[j], actions[j], rewards[j], values[j], logprobs[j])
					local_step += 1

				if terminated_state or (nan_occur is True):
					if (nan_occur is True):
						self.episodes[j].Pop()
					self.total_episodes.append(self.episodes[j])
					self.episodes[j] = EpisodeBuffer()

					self.env.Reset(True,j)

			if local_step >= self.buffer_size:
				break
				
			states = self.env.GetStates()
		
	def OptimizeSimulationNN(self):
		all_transitions = np.array(self.replay_buffer.buffer)
		for j in range(self.num_epochs):
			np.random.shuffle(all_transitions)
			for i in range(len(all_transitions)//self.batch_size):
				transitions = all_transitions[i*self.batch_size:(i+1)*self.batch_size]
				batch = Transition(*zip(*transitions))

				stack_s = np.vstack(batch.s).astype(np.float32)
				stack_a = np.vstack(batch.a).astype(np.float32)
				stack_lp = np.vstack(batch.logprob).astype(np.float32)
				stack_td = np.vstack(batch.TD).astype(np.float32)
				stack_gae = np.vstack(batch.GAE).astype(np.float32)
				
				a_dist,v = self.model(Tensor(stack_s))
				'''Critic Loss'''
				loss_critic = ((v-Tensor(stack_td)).pow(2)).mean()
				
				'''Actor Loss'''
				ratio = torch.exp(a_dist.log_prob(Tensor(stack_a))-Tensor(stack_lp))
				stack_gae = (stack_gae-stack_gae.mean())/(stack_gae.std()+ 1E-5)
				stack_gae = Tensor(stack_gae)
				surrogate1 = ratio * stack_gae
				surrogate2 = torch.clamp(ratio,min =1.0-self.clip_ratio,max=1.0+self.clip_ratio) * stack_gae
				loss_actor = - torch.min(surrogate1,surrogate2).mean()
				'''Entropy Loss'''
				loss_entropy = - self.w_entropy * a_dist.entropy().mean()

				self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
				self.loss_critic = loss_critic.cpu().detach().numpy().tolist()
				
				loss = loss_actor + loss_entropy + loss_critic

				self.optimizer.zero_grad()
				loss.backward(retain_graph=True)
				for param in self.model.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5,0.5)
				self.optimizer.step()
			print('Optimizing sim nn : {}/{}'.format(j+1,self.num_epochs),end='\r')
		print('')
	def OptimizeMuscleNN(self):
		muscle_transitions = np.array(self.muscle_buffer.buffer)
		for j in range(self.num_epochs_muscle):
			np.random.shuffle(muscle_transitions)
			for i in range(len(muscle_transitions)//self.muscle_batch_size):
				tuples = muscle_transitions[i*self.muscle_batch_size:(i+1)*self.muscle_batch_size]
				batch = MuscleTransition(*zip(*tuples))

				stack_JtA = np.vstack(batch.JtA).astype(np.float32)
				stack_tau_des = np.vstack(batch.tau_des).astype(np.float32)
				stack_L = np.vstack(batch.L).astype(np.float32)

				stack_L = stack_L.reshape(self.muscle_batch_size,self.num_action,self.num_muscles)
				stack_b = np.vstack(batch.b).astype(np.float32)

				stack_JtA = Tensor(stack_JtA)
				stack_tau_des = Tensor(stack_tau_des)
				stack_L = Tensor(stack_L)
				stack_b = Tensor(stack_b)

				activation = self.muscle_model(stack_JtA,stack_tau_des)
				tau = torch.einsum('ijk,ik->ij',(stack_L,activation)) + stack_b

				loss_reg = (activation).pow(2).mean()
				loss_target = (((tau-stack_tau_des)/100.0).pow(2)).mean()

				loss = 0.01*loss_reg + loss_target
				# loss = loss_target

				self.optimizer_muscle.zero_grad()
				loss.backward(retain_graph=True)
				for param in self.muscle_model.parameters():
					if param.grad is not None:
						param.grad.data.clamp_(-0.5,0.5)
				self.optimizer_muscle.step()

			print('Optimizing muscle nn : {}/{}'.format(j+1,self.num_epochs_muscle),end='\r')
		self.loss_muscle = loss.cpu().detach().numpy().tolist()
		print('')
	def OptimizeModel(self):
		self.ComputeTDandGAE()
		self.OptimizeSimulationNN()
		if self.use_muscle:
			self.OptimizeMuscleNN()
		
	def Train(self):
		self.GenerateTransitions()
		self.OptimizeModel()


	def Evaluate(self):
		self.num_evaluation = self.num_evaluation + 1
		h = int((time.time() - self.tic)//3600.0)
		m = int((time.time() - self.tic)//60.0)
		s = int((time.time() - self.tic))
		m = m - h*60
		s = int((time.time() - self.tic))
		s = s - h*3600 - m*60
		if self.num_episode is 0:
			self.num_episode = 1
		if self.num_tuple is 0:
			self.num_tuple = 1
		if self.max_return < self.sum_return/self.num_episode:
			self.max_return = self.sum_return/self.num_episode
			self.max_return_epoch = self.num_evaluation
		print('# {} === {}h:{}m:{}s ==='.format(self.num_evaluation,h,m,s))
		print('||Loss Actor               : {:.4f}'.format(self.loss_actor))
		print('||Loss Critic              : {:.4f}'.format(self.loss_critic))
		print('||Loss Muscle              : {:.4f}'.format(self.loss_muscle))
		print('||Noise                    : {:.3f}'.format(self.model.log_std.exp().mean()))		
		print('||Num Transition So far    : {}'.format(self.num_tuple_so_far))
		print('||Num Transition           : {}'.format(self.num_tuple))
		print('||Num Episode              : {}'.format(self.num_episode))
		print('||Avg Return per episode   : {:.3f}'.format(self.sum_return/self.num_episode))
		print('||Avg Reward per transition: {:.3f}'.format(self.sum_return/self.num_tuple))
		print('||Avg Step per episode     : {:.1f}'.format(self.num_tuple/self.num_episode))
		print('||Max Avg Retun So far     : {:.3f} at #{}'.format(self.max_return,self.max_return_epoch))
		self.rewards.append(self.sum_return/self.num_episode)
		
		self.SaveModel()
		
		print('=============================================')
		return np.array(self.rewards)

import matplotlib
import matplotlib.pyplot as plt

plt.ion()

i_epoch = 0

def Plot(y,title,num_fig=1,ylim=True):
	global i_epoch
	temp_y = np.zeros(y.shape)
	if y.shape[0]>5:
		temp_y[0] = y[0]
		temp_y[1] = 0.5*(y[0] + y[1])
		temp_y[2] = 0.3333*(y[0] + y[1] + y[2])
		temp_y[3] = 0.25*(y[0] + y[1] + y[2] + y[3])
		for i in range(4,y.shape[0]):
			temp_y[i] = np.sum(y[i-4:i+1])*0.2

	plt.figure(num_fig)
	plt.clf()
	plt.title(title)
	plt.plot(y,'b')
	
	plt.plot(temp_y,'r')

	plt.show()
	if ylim:
		plt.ylim([0,1])
	plt.pause(0.001)
	plt.savefig('reward.png', dpi=100)
	i_epoch+=1
	if i_epoch == 6000:
		plt.savefig('reward_6000.png', dpi=100)
	if i_epoch == 9000:
		plt.savefig('reward_9000.png', dpi=100)

import argparse
import os
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--model',help='model path')
	parser.add_argument('-d','--meta',help='meta file')

	args =parser.parse_args()
	if args.meta is None:
		print('Provide meta file')
		exit()

	ppo = PPO(args.meta)
	nn_dir = '../nn'
	if not os.path.exists(nn_dir):
	    os.makedirs(nn_dir)
	if args.model is not None:
		ppo.LoadModel(args.model)
	else:
		ppo.SaveModel()
	print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(),ppo.env.GetNumAction()))
	for i in range(ppo.max_iteration-5):
		ppo.Train()
		rewards = ppo.Evaluate()
		Plot(rewards,'reward',0,False)
		with open('rewards.txt', 'a') as f:
    			f.write(str(rewards[-1]))
    			f.write('\n')

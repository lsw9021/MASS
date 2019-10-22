import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
import time
from collections import OrderedDict
import numpy as np
from IPython import embed

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self,val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight)
		m.bias.data.zero_()
class MuscleNN(nn.Module):
	def __init__(self,num_total_muscle_related_dofs,num_dofs,num_muscles):
		super(MuscleNN,self).__init__()
		self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
		self.num_dofs = num_dofs
		self.num_muscles = num_muscles

		num_h1 = 1024
		num_h2 = 512
		num_h3 = 512
		self.fc = nn.Sequential(
			nn.Linear(num_total_muscle_related_dofs+num_dofs,num_h1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h1,num_h2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h2,num_h3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(num_h3,num_muscles),
			nn.Tanh(),
			nn.ReLU()		
		)
		self.std_muscle_tau = torch.zeros(self.num_total_muscle_related_dofs)
		self.std_tau = torch.zeros(self.num_dofs)

		for i in range(self.num_total_muscle_related_dofs):
			self.std_muscle_tau[i] = 200.0

		for i in range(self.num_dofs):
			self.std_tau[i] = 200.0
		if use_cuda:
			self.std_tau = self.std_tau.cuda()
			self.std_muscle_tau = self.std_muscle_tau.cuda()
			self.cuda()
		self.fc.apply(weights_init)
	def forward(self,muscle_tau,tau):
		muscle_tau = muscle_tau/self.std_muscle_tau

		tau = tau/self.std_tau
		out = self.fc.forward(torch.cat([muscle_tau,tau],dim=1))
		return out		

	def load(self,path):
		print('load muscle nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self,path):
		print('save muscle nn {}'.format(path))
		torch.save(self.state_dict(),path)
		
	def get_activation(self,muscle_tau,tau):
		act = self.forward(Tensor(muscle_tau.reshape(1,-1)),Tensor(tau.reshape(1,-1)))
		return act.cpu().detach().numpy()
		
class SimulationNN(nn.Module):
	def __init__(self,num_states,num_actions):
		super(SimulationNN,self).__init__()
		
		num_h1 = 256
		num_h2 = 256

		self.p_fc1 = nn.Linear(num_states,num_h1)
		self.p_fc2 = nn.Linear(num_h1,num_h2)
		self.p_fc3 = nn.Linear(num_h2,num_actions)
		self.log_std = nn.Parameter(torch.zeros(num_actions))

		self.v_fc1 = nn.Linear(num_states,num_h1)
		self.v_fc2 = nn.Linear(num_h1,num_h2)
		self.v_fc3 = nn.Linear(num_h2,1)

		# self.reward_container = Container(10000)

		torch.nn.init.xavier_uniform_(self.p_fc1.weight)
		torch.nn.init.xavier_uniform_(self.p_fc2.weight)
		torch.nn.init.xavier_uniform_(self.p_fc3.weight)

		self.p_fc1.bias.data.zero_()
		self.p_fc2.bias.data.zero_()
		self.p_fc3.bias.data.zero_()

		torch.nn.init.xavier_uniform_(self.v_fc1.weight)
		torch.nn.init.xavier_uniform_(self.v_fc2.weight)
		torch.nn.init.xavier_uniform_(self.v_fc3.weight)

		self.v_fc1.bias.data.zero_()
		self.v_fc2.bias.data.zero_()
		self.v_fc3.bias.data.zero_()

	def forward(self,x):
		p_out = F.relu(self.p_fc1(x))
		p_out = F.relu(self.p_fc2(p_out))
		p_out = self.p_fc3(p_out)

		p_out = MultiVariateNormal(p_out,self.log_std.exp())

		v_out = F.relu(self.v_fc1(x))
		v_out = F.relu(self.v_fc2(v_out))
		v_out = self.v_fc3(v_out)
		return p_out,v_out

	def load(self,path):
		print('load simulation nn {}'.format(path))
		self.load_state_dict(torch.load(path))

	def save(self,path):
		print('save simulation nn {}'.format(path))
		torch.save(self.state_dict(),path)
		
	def get_action(self,s):
		ts = torch.tensor(s)
		p,_ = self.forward(ts)
		return p.loc.cpu().detach().numpy()

	def get_random_action(self,s):
		ts = torch.tensor(s)
		p,_ = self.forward(ts)
		return p.sample().cpu().detach().numpy()

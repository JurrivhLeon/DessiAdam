"""
Neural Network and Deep Learning, Final Project.
Optimization.
Junyi Liao
An updated version of DessiLBI, with Adam.
Modified from https://github.com/DessiLBI2020/DessiLBI/blob/master/DessiLBI/code/slbi_opt.py.
"""

import torch
from torch.optim.optimizer import Optimizer, required
import copy
import math


"""
This is an updated version of DessiLBI.
Adam as an adaptive gradient method is used to update the W of DessiLBI 
in the process of optimizing the deep networks.
"""


class SLBI_Adam(Optimizer):
	def __init__(self, params, lr=required, kappa=1, mu=100, weight_decay=0, betas=(0.9, 0.999), eps=1e-8):
		defaults = dict(lr=lr, kappa=kappa, mu=mu, weight_decay=weight_decay, betas=betas, eps=eps)
		print('*******************************************')
		for key in defaults:
			print(key, ': ', defaults[key])
		print('*******************************************')
		super(SLBI_Adam, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(SLBI_Adam, self).__setstate__(state)

	def assign_name(self, name_list):
		for group in self.param_groups:
			for it, p in enumerate(group['params']):
				param_state = self.state[p]
				param_state['name'] = name_list[it]

	def initialize_slbi(self, layer_list=None):
		if layer_list is None:
			pass
		else:
			for group in self.param_groups:
				for p in group['params']:
					param_state = self.state[p]
					# State initialization
					param_state['step'] = 0
					# Exponential moving average of gradient values
					param_state['exp_avg'] = torch.zeros_like(p.data)
					# Exponential moving average of squared gradient values
					param_state['exp_avg_sq'] = torch.zeros_like(p.data)
					if param_state['name'] in layer_list:
						# Initialize V (as z_buffer) and Gamma.
						param_state['z_buffer'] = torch.zeros_like(p.data)
						param_state['gamma_buffer'] = torch.zeros_like(p.data)

	def step(self, closure=None):
		loss = None
		if closure is not None:
			loss = closure()
		for group in self.param_groups:
			mu = group['mu']
			kappa = group['kappa']
			lr_kappa = group['lr'] * group['kappa']
			lr_gamma = group['lr'] / mu
			weight_decay = group['weight_decay']
			beta1, beta2 = group['betas']
			eps = group['eps']
			for p in group['params']:
				if p.grad is None:
					continue
				d_p = p.grad.data
				param_state = self.state[p]

				exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
				param_state['step'] += 1

				# if momentum != 0:
				# 	if 'momentum_buffer' not in param_state:
				# 		buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
				# 		buf.mul_(momentum).add_(d_p)
				# 	# Use momentum.
				# 	else:
				# 		buf = param_state['momentum_buffer']
				# 		buf.mul_(momentum).add_(1 - dampening, d_p)
				# 	d_p = buf

				if weight_decay != 0 and len(p.data.size()) != 1 and 'bn' not in param_state['name']:
					# Weight decay.
					d_p.add_(weight_decay, p.data)

				if 'z_buffer' in param_state:
					# L_modified = Loss + FrobeniusNorm(W - Gamma)^2 / (2 * mu)
					# Update the weight: W_{k + 1} = W_k - lr_kappa * Grad_W(L_modified).
					d_p = d_p + (p.data - param_state['gamma_buffer']) / mu
					# new_grad = d_p * lr_kappa + (p.data - param_state['gamma_buffer']) * lr_kappa / mu
					last_p = copy.deepcopy(p.data)
					# p.data.add_(-new_grad)
					# Update V: V_{k + 1} = V_k - lr_gamma * Grad_Gamma(L_modified).
					param_state['z_buffer'].add_(-lr_gamma, param_state['gamma_buffer'] - last_p)
					# Update Gamma: Gamma_{k + 1} = kappa * Prox(V_{k + 1}).
					if len(p.data.size()) == 2:
						param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], 1)
					elif len(p.data.size()) == 4:
						param_state['gamma_buffer'] = kappa * self.shrink_group(param_state['z_buffer'])
					else:
						pass
				"""
				else: p.data.add_(-lr_kappa, d_p)  # for bias update as vanilla sgd.
				"""
				# Decay the first and second moment running average coefficient
				exp_avg.mul_(beta1).add_(1 - beta1, d_p)
				exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, d_p, d_p)
				denom = exp_avg_sq.sqrt().add_(eps)
				bias_correction1 = 1 - beta1 ** param_state['step']
				bias_correction2 = 1 - beta2 ** param_state['step']
				step_size = lr_kappa * math.sqrt(bias_correction2) / math.sqrt(bias_correction1)
				p.data.addcdiv_(-step_size, exp_avg, denom)

	def calculate_w_star_by_layer(self, layer_name):
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if 'z_buffer' in param_state and param_state['name'] == layer_name:
					if len(p.data.size()) == 2:
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					elif len(p.data.size()) == 4:
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					else:
						pass
				else:
					pass

	def calculate_all_w_star(self):
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if 'z_buffer' in param_state:
					if len(p.data.size()) == 2:
						# print(p.data.size())
						# print(param_state['gamma_buffer'].size())
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					elif len(p.data.size()) == 4:
						# print(p.data.size())
						# print(param_state['gamma_buffer'].size())
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					else:
						pass

	def calculate_layer_residue(self, layer_name):
		diff = 0
		for group in self.param_groups:
			mu = group['mu']
			for p in group['params']:
				param_state = self.state[p]
				if param_state['name'] == layer_name:
					if 'gamma_buffer' in param_state:
						diff = ((p.data - param_state['gamma_buffer']) * (p.data - param_state['gamma_buffer'])).sum().item()
					else:
						pass
		diff /= (2 * mu)
		print('Residue of' + layer_name + ': ', diff)

	def calculate_all_residue(self):
		diff = 0
		for group in self.param_groups:
			mu = group['mu']
			for p in group['params']:
				param_state = self.state[p]
				if 'gamma_buffer' in param_state:
					diff += ((p.data - param_state['gamma_buffer']) * (p.data - param_state['gamma_buffer'])).sum().item()
		diff /= (2 * mu)
		print('Residue: ', diff)

	@staticmethod
	def shrink(s_t, lam):
		# proximal mapping for 2-d weight(fc layer)
		gamma_t = s_t.sign() * (torch.max(s_t.abs() - (lam * torch.ones_like(s_t)), torch.zeros_like(s_t)))
		return gamma_t

	@staticmethod
	def shrink_group(ts):
		# shrinkage for 4-d weight(conv layer)
		ts_reshape = torch.reshape(ts, (ts.shape[0], -1))
		ts_norm = torch.norm(ts_reshape, 2, 1)
		ts_shrink = torch.max(torch.zeros_like(ts_norm), torch.ones_like(ts_norm) - torch.div(torch.ones_like(ts_norm), ts_norm))
		ts_return = torch.transpose(torch.mul(torch.transpose(ts_reshape, 0, 1), ts_shrink), 0, 1)
		ts_return = torch.reshape(ts_return, ts.shape)
		return ts_return

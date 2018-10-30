import numpy as np
import torch




def pendulum(t, A0, delta0, k, b, m):
	"""
	Solution x(t) for pendulum differential equation
		mx'' = -kx + bx'
	Returns position at time t

	Parameters:
		- t: time
		- A0: starting amplitude
		- delta0: phase
		- k: spring constant
		- b: damping factor
	"""
	A = 1 - b**2 / (4 * m * k)
	if A < 0:
		return None
	w = np.sqrt(k/m)* np.sqrt(A)
	result = A0 * np.exp( - t * b / (2. * m) ) * np.cos(w * t + delta0)
	return result

def target_loss(pred,answer):
	"""
	
	"""
	pred = pred[:,0]
	
	return torch.mean(torch.sum((pred - answer)**2), dim=0)
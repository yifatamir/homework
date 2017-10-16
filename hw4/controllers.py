import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.env = env

	def get_action(self, state):
		""" Your code should randomly sample an action uniformly from the action space """
		""" YOUR CODE HERE """
		return self.env.action_space.sample()

class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" Note: be careful to batch your simulations through the model for speed """
		""" – To evaluate the costs of imaginary rollouts, use trajectory_cost_fn, which
		requires a per-timestep cost_fn as an argument. Notice that the MPC controller
		gets a cost function as a keyword argument—this is what you should use!
		– When generating the imaginary rollouts starting from a state s, be efficient and
		batch the computation. At the first step, broadcast s to have shape (number of
		fictional rollouts, observation dim), and then use that as an input to the dynamics
		model prediction to produce the batch of next steps.
		– The cost functions are also designed for batch computations, so you can feed the
		whole batch of trajectories at once to trajectory_cost_fn. For details on
		how, read the code. """
		"""You shouldn't be using the environment for MPCcontroller's get_action.  
		You should use the dynamics model to predict the next state, instead of using the env 
		to get the actual next state."""
		""" YOUR CODE HERE """
		states = np.repeat(np.array([[state]]), self.num_simulated_paths, axis = 0)
		actions = np.array([[self.env.action_space.sample()] for i in range(self.num_simulated_paths)])
		next_s = np.array([[state] for state in self.dyn_model.predict(states[:, -1, :], actions[:, -1, :])])
		states = np.concatenate((states, next_s), axis = 1)
		for t in range(self.horizon-1):
			actions = np.concatenate((actions, np.array([[self.env.action_space.sample()] for i in range(self.num_simulated_paths)])), axis = 1)
			next_s = np.array([[state] for state in self.dyn_model.predict(states[:, -1, :], actions[:, -1, :])])
			states = np.concatenate((states, next_s), axis = 1)
		cost_dict = {i : trajectory_cost_fn(self.cost_fn, states[i,:-1,:], actions[i, :, :], states[i, 1:, :]) for i in range(self.num_simulated_paths)}
		j_star = min(cost_dict, key=cost_dict.get)
		return actions[j_star,0,:]
		






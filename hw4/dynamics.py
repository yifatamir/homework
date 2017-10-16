import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ Note: Be careful about normalization """
        """ Pay careful attention to the keyword args for the dynamics model. The normalization
        vectors are inputs here, and you need these for normalizing inputs and
        denormalizing outputs from the model.
        – You want the neural network for your dynamics model to output differences in
        states, instead of outputting next states directly. Then using the estimated state
        difference ∆ and the current state ˆ s, you will predict the estimated next state. """
        """ YOUR CODE HERE """
        self.env = env
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        self.normalization = normalization
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess

        # get observation dimensions and actions dimensions
        obs_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]

        # create tf model placeholders
        states = tf.placeholder(tf.float32, [None, obs_dim])
        actions = tf.placeholder(tf.float32, [None, ac_dim])
        true_next_states = tf.placeholder(tf.float32, [None, obs_dim])

        # normalize states and actions and then concatenate them together
        states_norm = self.normalize(states, normalization[0], normalization[1])
        actions_norm = self.normalize(actions, normalization[4], normalization[5])
        state_action_pairs = tf.concat([states_norm, actions_norm], 1)

        # get predicted state differences, denormalize, and then convert to predicted next states
        delta_predictor = build_mlp(state_action_pairs, obs_dim, "dynamics", n_layers=n_layers, size=size, activation=activation, output_activation=output_activation)
        delta_predictor = normalization[2] + normalization[3] * delta_predictor
        self.predicted_states = states + delta_predictor

        # compute loss function and update operation
        self.loss_fn = tf.losses.mean_squared_error(true_next_states, predicted_states)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_fn)

    def normalize(self, tensor, mu, sigma):
        return tf.divide(tf.subtract(tensor, mu), tf.add(sigma, 0.000000001))

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit 
        the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        Use the AdamOptimizer to train the dynamics model. For details on how many steps
        of gradient descent to take, we recommend that you study the experimental details in
        (Nagabandi, 2017).
        – How to use the normalization statistics: given a state s and an action a, and
        normalization statistics µs, σs, µa, σa, µ∆, σ∆, you want your network to compute an 
        estimate of the state difference ∆. ___ is an elementwise vector multiply and  is a small positive value (to
        prevent divide-by-zero).
        """
        """YOUR CODE HERE """
        obs = np.array([data[i]["observations"] for i in range(len(data))])
        next_obs = np.array([data[i]["next_observations"] for i in range(len(data))])
        acts = np.array([data[i]["actions"] for i in range(len(data))])

        for i in range(self.iterations):
            obs_i, next_obs_i, acts_i = shuffle(obs, next_obs, acts, n_samples=self.batch_size)
            _, l = self.sess.run([self.update_op, self.loss_fn], feed_dict = {states:obs_i, actions:acts_i, true_next_states:next_obs_i})

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) 
        next states as predicted by using the model """
        """ YOUR CODE HERE """
        return self.sess.run([self.predicted_states], feed_dict = {states:states, actions:actions})



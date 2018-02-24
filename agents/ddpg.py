""" DDPG agent """
import numpy as np
import os
import pandas as pd
import h5py

from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl import util
from quad_controller_rl.agents.actor import Actor
from quad_controller_rl.agents.critic import Critic
from quad_controller_rl.agents.replay_buffer import ReplayBuffer
from quad_controller_rl.agents.ounoise import OUNoise

class DDPG(BaseAgent):
    """Sample agent that searches for optimal policy randomly."""

    def setup_weights(self):
        # save weights
        self.load_weights = True
        self.save_weights_every = 50
        self.model_dir = util.get_param('out')
        self.model_name = "ddpg"
        self.model_ext = ".h5"
        if self.load_weights or self.save_weights_every:
            self.actor_filename = os.path.join(self.model_dir,
                    "{}_actor{}".format(self.model_name, self.model_ext))
            self.critic_filename = os.path.join(self.model_dir,
                    "{}_critic{}".format(self.model_name, self.model_ext))
            print("Actor filename :", self.actor_filename)
            print("Critic filename:", self.critic_filename)
        if self.load_weights and os.path.isfile(self.actor_filename):
            try:
                self.actor_local.model.load_weights(self.actor_filename)
                self.critic_local.model.load_weights(self.critic_filename)
                print("Model weights loaded from file!")
            except Exception as e:
                print("Unable to load model weights from file!")
                print("{}: {}".format(e.__class__.__name__, str(e)))
        else:
            self.critic_target.set_weights(self.critic_local)
            self.actor_target.set_weights(self.actor_local)

    def __init__(self, task):
        self.task = task
        self.state_size = 3
        self.action_size = 3

        #set action space limits
        self.action_low = self.task.action_space.low[0:3]
        self.action_high = self.task.action_space.high[0:3]
        print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
            self.task.observation_space.shape, self.task.action_space.shape,
            self.state_size, self.action_size))

        action = [self.action_size, self.action_low, self.action_high]

        #Initialize network
        #Actor
        self.actor_local = Actor(self.state_size, action)
        self.actor_target = Actor(self.state_size, action)
        #Critic
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.setup_weights()

        #noise
        self.noise = OUNoise(self.action_size)

        #Replay buffer
        self.buffer_size = 100000
        self.batch_size = 128 
        self.memory = ReplayBuffer(self.buffer_size)

        #Hyper params
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

        # log file
        self.stats = os.path.join(util.get_param('out'), "stats_{}.csv".format(
          util.get_timestamp()))
        self.episode_no = 1
        self.stats_columns = ['episodes', 'total_reward']
        print("Saving stats {} to {}".format(self.stats_columns, self.stats))


        # Episode variables
        self.reset_episode_vars()

    def preprocess_state(self, state):
        return state[0:3]

    def postprocess_action(self, action):
        constrained_action = np.zeros(self.task.action_space.shape)
        constrained_action[0:3] = action
        return constrained_action

    def write(self, data):
        df_stats = pd.DataFrame([data], columns=self.stats_columns)
        df_stats.to_csv(self.stats, mode='a', index=False,
         header=not os.path.isfile(self.stats))

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0

    def step(self, state, reward, done):

        state = self.preprocess_state(state)
        #choose an action
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.count += 1
            self.memory.add_experience(state, action, reward, self.last_state, done)

        # Learn, if at end of episode
        if self.memory.len() > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

            self.episode_no += 1
        if done:
            if self.save_weights_every and self.episode_no % self.save_weights_every == 0:
                self.actor_local.model.save_weights(self.actor_filename)
                self.critic_local.model.save_weights(self.critic_filename)
                print("Model weights saved at episode", self.episode_no)
            self.write([self.episode_no, self.total_reward])
            self.reset_episode_vars()

        self.last_state = state
        self.last_action = action
        return self.postprocess_action(action)

    def act(self, state):
        # Choose action based on given state and policy
        states = np.reshape(state, [-1, self.state_size])
        actions = self.actor_local.predict(states)

        return actions + self.noise.sample()

    def learn(self, experiences):
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(
                np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(
                np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(
                np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.state_next for e in experiences if e is not None])

        # Get predicted next states and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch(
                [next_states, actions_next])

        #compute Q targets 
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        #train actor model
        action_gradients = np.reshape(self.critic_local.get_action_gradients(
            [states, actions, 0]),(-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        #update
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)


    def soft_update(self, local_model, target_model):
        '''update model params'''
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.update_weights(new_weights)

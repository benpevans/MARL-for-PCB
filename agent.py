import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import numpy as np
import timeit

class ReplayBuffer():
    def __init__(self, mem_size, input_dims, partial_obs=None):
        self.mem_size = mem_size
        self.mem_counter = 0
        self.partial_obs = partial_obs
        self.obs_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.next_obs_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)
        if self.partial_obs:
            self.x_y_memory = np.zeros((self.mem_size, 2), dtype=np.float32)
            self.next_x_y_memory = np.zeros((self.mem_size, 2 ), dtype=np.float32)
    
    def store(self, state, next_state, action, reward, game_over):
        i = self.mem_counter % self.mem_size
        if self.partial_obs:
            self.x_y_memory[i] = state[1]
            self.next_x_y_memory[i] = next_state[1]
            self.obs_memory[i] = state[0]
            self.next_obs_memory[i] = next_state[0]
        else:
            self.obs_memory[i] = state
            self.next_obs_memory[i] = next_state
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.terminal_memory[i] = 1 - int(game_over)
        self.mem_counter += 1
    
    def sample(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        if self.partial_obs:
            states = np.expand_dims(self.obs_memory[batch], -1), self.x_y_memory[batch]
            next_states = np.expand_dims(self.next_obs_memory[batch], -1), self.next_x_y_memory[batch]
        else:
            states = self.obs_memory[batch]
            next_states = self.next_obs_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]
        return states, next_states, actions, rewards, terminals

class NN():
    def __init__(
        self, 
        learning_rate, 
        inputshape,
        n_actions=8,
        partial_obs=None,
        load=None):

        self.learning_rate = learning_rate
        self.inputshape = inputshape
        self.n_actions = n_actions
        self.partial_obs = partial_obs
        
        if not load:
            self.qnet = self._q_network()
        else:
            self.qnet = self._load(load)


    def _q_network(self):
        """
        Creates a q_network with 3 hidden layers:
        1 convolutional layer with 32 filters and a kernel size of 3
        1 convolutional layer with 64 filers and a kernel size of 1
        1 dense fully connected layer with 512 nodes
        """
        obs_input = keras.Input(shape=self.inputshape)
        conv1 = layers.Conv2D(32, 3, activation="relu")(obs_input)
        conv2 = layers.Conv2D(64, 1, activation ="relu")(conv1)
        conv_output = layers.Flatten()(conv2)
        initial_out = conv_output
        if self.partial_obs:
            x_y_input = keras.Input(shape=(2,))
            concat = layers.Concatenate()([conv_output, x_y_input])
            initial_out = concat
        dense = layers.Dense(512, activation="relu")(initial_out)
        action_output = layers.Dense(self.n_actions, activation="linear")(dense)
        if self.partial_obs:
            model = keras.Model(inputs=[obs_input, x_y_input], outputs=action_output)
        else:
            model = keras.Model(inputs=obs_input, outputs=action_output)
        return model

    def save(self, fname):
        self.qnet.save(fname)

    def _load(self, fname):
        return keras.models.load_model(fname)

class Agent():
    def __init__(
        self, 
        learning_rate, 
        gamma, 
        n_actions, 
        epsilon, 
        batch_size,
        input_dims, 
        epsilon_dec=1e-3, 
        epsilon_end=0.01,
        mem_size=100000, 
        fname='one_path_model.h5', 
        load_model=False, 
        partial_obs=None):

        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.eps_min = epsilon_end
        self.epsilon_dec = epsilon_dec
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_file = fname
        self.partial_obs = partial_obs
        
        if partial_obs:
            input_dims = partial_obs*2+1, partial_obs*2+1
        self.memory = ReplayBuffer(mem_size, input_dims, partial_obs)
        self.optimiser = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        
        a,b = input_dims
        qnet_input_dims = (a,b,1)
        if load_model:
            self.qnet = NN(learning_rate, qnet_input_dims, n_actions, partial_obs, fname)
            self.target_qnet = NN(learning_rate, qnet_input_dims, n_actions, partial_obs, fname)
        else:
            self.qnet = NN(learning_rate, qnet_input_dims, n_actions, partial_obs)
            self.target_qnet = NN(learning_rate, qnet_input_dims, n_actions, partial_obs)

    def store(self, state, nextstate, action, reward, terminal):
        self.memory.store(state, nextstate, action, reward, terminal)
    
    def choose_action(self, obs):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            if self.partial_obs:
                obs_tensor = np.expand_dims(np.expand_dims(obs[0], 0),-1)
                obs_tensor = (obs_tensor, np.expand_dims(obs[1], 0))     
            else:
                obs_tensor = np.expand_dims(np.expand_dims(obs, 0), -1)
            actions = self.qnet.qnet(obs_tensor)
            action = np.argmax(actions)
        return action
    
    def learn(self):
        if self.memory.mem_size < self.batch_size or self.memory.mem_counter % 8 != 0:
            return
        states, next_states, actions, rewards, terminals = \
             self.memory.sample(self.batch_size)

        # calculate updated q-values
        NewQValues = rewards + self.gamma * tf.reduce_max(self.target_qnet.qnet(next_states), axis=1) * terminals

        # masks to calculate the loss on the updated Q-values
        masks = tf.one_hot(actions, self.n_actions)

        with tf.GradientTape() as tape:

            # Train model on the states and updated q-values & apply masks to the Q-values
            QValues = tf.reduce_sum(tf.multiply(self.qnet.qnet(states), masks), axis=1)

            #calculate the huber loss
            loss = keras.losses.Huber()(NewQValues, QValues)

        grads = tape.gradient(loss, self.qnet.qnet.trainable_variables)
        self.optimiser.apply_gradients(zip(grads, self.qnet.qnet.trainable_variables))

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > \
            self.eps_min else self.eps_min

        # update the target q network
        if self.memory.mem_counter % 5000 == 0:
            self.target_qnet.qnet.set_weights(self.qnet.qnet.get_weights())

    def save_model(self):
        self.qnet.save(self.model_file)
    
    def load_model(self):
        return keras.models.load_model(self.model_file)

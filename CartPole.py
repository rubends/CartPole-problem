import gym

import numpy
import random
from time import sleep
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model


class DqnAgent:

    def __init__(self, env, training):
        self.env = env
        self.training = training

        # constants
        self.learning_rate = 0.001 # how much neural net learns in each iteration, OG = 0.5, weight, best small at start, larger to end but only 100 iterations
        self.gamma = 0.94 # discount rate = calculate the future discounted reward
        self.epsilon = 1.0 # exploration rate = rate that agent randomly decides its action rather than prediction 
        # if epsilon starts as 0.1, it will starts to predict in 90% of the time, which is way too little in
        # early phase of the net, so epsilon start very big and get smaller every iteration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = list()
        self.batch_size = 32
        
        # create neural network
        state_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.n
        self.model = Sequential()
        # First input layer with a hidden layer of nodes
        self.model.add(Dense(16, input_dim=state_shape[0], activation="relu"))
        # after the first layer, the hidden layers:
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(16, activation="relu"))
        self.model.add(Dense(action_shape))
        self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))

    # randomly select its action by epsilon
    def act(self, state):
        # epsilon-greedy = random action in epsilon percentage of the time
        rnd = numpy.random.rand()
        if rnd <= self.epsilon and self.training:
            return self.env.action_space.sample()
        
        # pick reward by state
        act_values = self.model.predict(state)
        # pick action by reward
        return numpy.argmax(act_values[0])

    # keep hold of array of experiences 
    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    # trains the neural net with experiences in memory
    def replay(self):
        if len(self.memory) < self.batch_size: #can't sample more than there are items in list
            batch = len(self.memory)
        else:
            batch = self.batch_size

        experience = random.sample(self.memory, batch)
        states, targets_f = [], []

        for state, action, reward, new_state, done in experience:
            if done:
                target = reward
            else:
                # predict future reward: Q function
                target = reward + (self.gamma * numpy.amax(self.model.predict(new_state)[0]))
    
            # map current state to future reward
            target_f = self.model.predict(state)
            target_f[0][action] = target

            states.append(state[0])
            targets_f.append(target_f[0])

            #train the net
        self.model.fit(numpy.array(states), numpy.array(targets_f), batch_size=self.batch_size, epochs=1, verbose=0)
        
        # adapt epsilon to stage in training
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # * ipv - to start big and end smaller

    def save_model(self, name):
        self.model.save(name)

    def load_model(self, name):
        self.model = load_model(name)


def run_agent(env, training=False, number_of_episodes=100, model_name=None):
    total_reward = 0
    agent = DqnAgent(env, training)

    if not training:
        try:
            if model_name is None:
                agent.load_model("{}.model".format(env.spec.id.lower()))
            else:
                agent.load_model(model_name)
        except:
            print("Failed to load {}".format(env.spec.id.lower()))
            return

    for episode in range(number_of_episodes):
        done = False
        total_episode_reward = 0
        state = env.reset()
        state = numpy.reshape(state, [1, 4])

        while not done:
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)

            new_state = numpy.reshape(new_state, [1, 4])

            agent.remember(state=state,
                           action=action,
                           reward=reward,
                           new_state=new_state,
                           done=done)

            agent.replay()

            env.render()
            if not training:
                sleep(0.02)
            state = new_state
            total_episode_reward += reward


        print("Total reward for episode {} is {}".format(episode, total_episode_reward))
        total_reward += total_episode_reward

    if training:
        agent.save_model("{}.model".format(env.spec.id.lower()))
        print("Total training reward for agent after {} episodes is {}".format(number_of_episodes, total_reward))
    else:
        print("Result of {} = {}".format(env.spec.id, total_reward))


def main():
    env = gym.make("CartPole-v1")

    # Train the agent
    run_agent(env, training=True, number_of_episodes=100)

    # Test performance of the agent
    run_agent(env, training=False, number_of_episodes=10)

    # Demo
    # run_agent(env, training=False, number_of_episodes=10, model_name="cartpole-v1.model")


if __name__ == "__main__":
    main()

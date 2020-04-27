import gym
import numpy as np

env = gym.make('FrozenLake-v0')
env.reset()

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

LEARNING_RATE = .8
DISCOUNT = .95
num_episodes = 2000
SHOW_EVERY = 20

rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    state = env.reset()
    done = False

    if i % SHOW_EVERY ==0:
        print(i)
        render = True
    else:
        render = False

    #The Q-Table learning algorithm
    while not done:
        #Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        new_state,reward,done,_ = env.step(action)

        

        if render:
            env.render()
        if not done:
            max_futurn_q = np.max(Q[new_state, :])
            current_q = Q[state,action]

        

        #Update Q-Table with new knowledge
        Q[state,action] = current_q + LEARNING_RATE * (reward + DISCOUNT * max_futurn_q - current_q)
        state = new_state
        if done:
            break

print ("Final Q-Table Values")
print (Q)
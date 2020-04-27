import gym
import numpy as np
env = gym.make('MountainCar-v0')
env.reset()


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2001

SHOW_EVERY = 20

DISCREATE_OS_SIZE = [20]*len(env.observation_space.high)
discreate_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCREATE_OS_SIZE

# print(discreate_os_win_size)
q_table = np.random.uniform(low=-2, high=0, size=(DISCREATE_OS_SIZE + [env.action_space.n]))

def get_discreate_state(state):
    discrete_state = (state - env.observation_space.low) / discreate_os_win_size
    
    state = discrete_state.astype(np.int)
    return (state[0], state[1])

for episode in range(EPISODES):

    if episode % SHOW_EVERY ==0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discreate_state(env.reset())
    done = False

    while not done:
        action = np.argmax(q_table[discrete_state]) # 0-left 1-nothing 2-right 
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discreate_state(new_state)
        if render:
            env.render()
        if not done:
            max_futurn_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE * (reward + DISCOUNT*max_futurn_q)    
            # Q_new = (1-alpha)*Q_old + alpha*(reward + discount*max(Q))

            q_table[discrete_state+(action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0 

        discrete_state = new_discrete_state
    

env.close()
 

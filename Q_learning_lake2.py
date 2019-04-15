# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:04:28 2019

@author: DMa
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:14:14 2019

@author: DMa
"""

import gym 
import numpy as np
import time

from gym.envs.registration import register

#register(
#    id='Deterministic-8x8-FrozenLake-v0',
#    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
#    kwargs={'map_name': '8x8', 'is_slippery': False}
#)

#env = gym.make('FrozenLake-v0')
env = gym.make('Deterministic-8x8-FrozenLake-v0')

#print(env.observation_space.n)

def Q_learning(env, epsilon, lr, gamma, episodes):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q=np.zeros((n_states, n_actions))
    
    for episode in range(episodes):
        state=env.reset()
        terminate = False
        while True:
            action=epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminate, info=env.step(action)
            
            if reward ==0:
                if terminate:
                    r=-5
                    Q[next_state, :]=np.ones(n_actions)*r
                else:
                    r=-1
            if reward==1:
                r=100
                Q[next_state, :]=np.ones(n_actions)*r
            Q[state,action] = Q[state,action] + lr * (r + gamma * np.max(Q[next_state, :]) - Q[state, action])

            state=next_state
            if terminate:
                break

    return Q

def epsilon_greedy(Q, state, epsilon):  
    p=np.random.uniform()
    if p<epsilon:
        action_prime=env.action_space.sample()
    else:
        action_prime=np.argmax(Q[state,:])
    return action_prime

def QLearning_process(env, Q, max_steps=100):
    state=env.reset()
    i=0
    while i<max_steps:
        action=np.argmax(Q[state,:])
        next_state, reward, terminate, info=env.step(action)
        print('step ', i, ':')
        env.render()
        state=next_state
        i+=1
        if terminate:
            print('fall into hole')
            break


epsilon=0.1
lr=0.8
gamma=0.8
episodes=10000
#Q=Q_learning(env, epsilon, lr, gamma, episodes)

allTime=[]
for i in range(100):
    start = time.perf_counter() 
    Q=Q_learning(env, epsilon, lr, gamma, episodes)
    allTime.append(time.perf_counter() -start)
allTime.sort()   
print (sum(allTime[0:3])/3.)
print('Q_learning finished')
print(Q)
#QLearning_process(env, Q)
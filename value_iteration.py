# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 18:10:45 2019

@author: DMa
"""

import gym 
import numpy as np
import time


def getReward(env):
    n_s, n_a = env.observation_space.n, env.action_space.n
    R=np.zeros((n_s, n_a))
    
    for s in range(n_s):
        for a, moves in env.env.P[s].items():
            for move in moves:
                R[s,a] += move[2]*move[0]
    return R

def getT(env):
    n_s, n_a = env.observation_space.n, env.action_space.n
    P=np.zeros((n_s, n_a, n_s))
    
    for s in range(n_s):
        for a, moves in env.env.P[s].items():
            for move in moves:
                P[s,a,move[1]] += move[0]
    return P

def policy_evaluation(pi, P, R, gamma, n_s):
    n_s, n_a = env.observation_space.n, env.action_space.n
    p=np.zeros((n_s, n_s))
    r=np.zeros((n_s, 1))
    
    for s in range(n_s):
        r[s]=R[s, pi[s]]
        p[s, :]=P[s, pi[s], :]
        V= np.linalg.inv((np.eye(n_s) - gamma * p)).dot(r)[:, 0]
    return V

def policy_iteration(env, epsilon, gamma, max_iter=10000):
    n_s, n_a = env.observation_space.n, env.action_space.n
    V=np.zeros(n_s)
    pi=np.ones(n_s, dtype=int)
    
    R=getReward(env)
    P=getT(env)
    
    i=0
    while i < max_iter:
        V_prev=V.copy()
        V=policy_evaluation(pi, P, R, gamma, n_s)
        for s in range(n_s):
            pi[s]=np.argmax(R[s,:]+gamma*P[s, :, :].dot(V))
            
        i +=1
        
        if np.linalg.norm(V_prev - V)<epsilon:
            print("policy iteration converges at iteration", i)
            break
    return V, pi

def print_value(V, width=8, height=8):
    return np.around(np.resize(V, (width, height)), 4)

def print_policy(V, width=8, height=8):
    table = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    policy = np.resize(V, (width, height))

    return np.vectorize(table.get)(policy)

def value_iteration(env, epsilon, gamma, max_iter=10000):
    n_s, n_a = env.observation_space.n, env.action_space.n
    V = np.zeros(n_s)    
    R = getReward(env)
    P = getT(env)
    
    i = 0
    while i < max_iter:
        i += 1
        prev_V = V.copy()
        for s in range(n_s):
            V[s] = max(R[s,:] + gamma * P[s, :, :].dot(V))
        i+=1
        
        if np.linalg.norm(prev_V - V) <= epsilon:
            print("Value iteration converged at iteration ", i)
            break
    
    return V

def value_to_policy(env, gamma, V):
    n_s, n_a = env.observation_space.n, env.action_space.n
    
    pi = np.zeros(n_s, dtype=int)
    for s in range(n_s):
        best_action = 0
        best_reward = -float("inf")
        for a in range(n_a):
            moves = env.env.P[s][a] # [(prob, next_state, reward, terminate), ...]
            avg_reward = sum([prob * reward + gamma * V[next_state] for (prob, next_state, reward, info) in moves])
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = a
        
        pi[s] = best_action
    
    return pi

def plot_process(env, pi, max_steps=1000):
    state=env.reset()
    env.render()
    i=0
    action=pi[state]
    while i<max_steps:        
        next_state, reward, terminate, info=env.step(action)
        env.render()
        action=pi[next_state]
        i+=1
        if terminate:
            break

env = gym.make('FrozenLake8x8-v0')
#n_s=len(env.env.P)
#n_a=len(env.env.P[0])
#print(n_s)
#print(n_a)
#print(env.P[0].items())
epsilon=1e-8
gamma=0.8
max_iter=10000
allTime=[]
for i in range(100):
    start = time.perf_counter() 
    V=value_iteration(env, epsilon, gamma, max_iter)
    pi=value_to_policy(env, gamma, V)
    allTime.append(time.perf_counter() -start)
allTime.sort()    
#print('time used: ', time.perf_counter() -start, 's')
print("state values:")
print(print_value(V))
print("policy:")
print(print_policy(pi))
#plot_process(env, pi)


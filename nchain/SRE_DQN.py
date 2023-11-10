#IMPORTS
import gym

#Importing Environment
from nchain_env import NChainEnv

import numpy as np
import random

import torch
import numpy as np
from collections import deque, namedtuple

#import learning_agents
import torch.optim as optim

import torch
import torch.nn as nn  
import torch.nn.functional as F

#Importing DQN Networks
import Q_Networks
from Q_Networks import QNetwork_DQN, QNetwork_DQNHER

#Importing Replay Buffers
from Experience_Replays import ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#BUFFER_SIZE = 100000 # replay buffer size
#BATCH_SIZE = 32        # minibatch size
#GAMMA = 0.95           # discount factor
#LR = 1e-3              # learning rate 
#UPDATE_EVERY = 5        # how often to update the network (When Q target is present)

from Agents import Agent_DQNHER


def dqn_sre(reward_target,n_episodes=100000, max_t= 500, eps_start=1.0, eps_end=0.1, eps_decay=0.99995):
    print(pow(eps_decay,n_episodes))

    scores = []                 # list containing scores from each episode
    scores_window_printing = deque(maxlen=10) # For printing in the graph
    scores_window= deque(maxlen=100)  # last 20 scores for checking if the avg is more than 195
    eps = eps_start                    # initialize epsilon
    
    hindsight_goals = []
    foresight_goals = []
    hindsight_enable = False
    foresight_enable = False
    
    #Check if agent learns to solve a cube that is one move away from goal state
    
    
    for i_episode in range(1, n_episodes+1):
        
        state = env.reset()
        score = 0
        
        traj_val = []
         
        # Run for one trajectory
        for t in range(max_t):
            
            #We store all we need for each trajectory
            
            #Choosing an action
            action = agent.act(np.concatenate((state,fin_goal)), eps)

            #Executing that action
            next_state, reward, done, _ = env.step(action)
             
            #print(state)
            traj_val.append([state,action,reward,next_state,done])
            
            #print('traj_val',traj_val)
            #print('len',len(traj_val))
            #agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break 
                
        # Adding Hindsight Goals
        
        psuedo_goal = next_state
        
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(hindsight_goals) == 0:
                hindsight_goals.append(psuedo_goal)
                
            for hind_goal in hindsight_goals:
                if ((state == hind_goal).all()):
                    flag = 1
                    
            if flag == 0:
                hindsight_goals.append(psuedo_goal)
        #print(len(traj_val))
        
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],psuedo_goal))
            new_next_state = np.concatenate((sublist[3],psuedo_goal))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]
            
            if ((sublist[3] == psuedo_goal).all() and reward < 0):
                    #print('sublist[3]',sublist[3])
                    #print('hind_goal_in_check',hind_goal)
                    #print('entering check',(sublist[3] == hind_goal).all())
                    reward = 100
                    
            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
        
        #For scramble trajectories
        scr_done = False
        scramble_length = 25
        
        #Start a new environment from the goal to generate samples

        if i_episode % 25 == 1:
            new_env_scr.init_state(fin_goal)
        #new_env_scr.init_state(fin_goal)
        scr_state = new_env_scr.get_state()
        #print('fin_goal',fin_goal)
        
        for i in range(scramble_length):
            
            new_env_scr.init_state(scr_state)
            
            #Scrambler Part
            scramble_action = random.choice(np.arange(0,2))
            
            #if i == 0:
                #scramble_action = int(3+ 2*random.random())
            #else:
                #scramble_action = agent.scramble_action(curr_state,eps)
                
            temps_state, _,  temp_done, _ = new_env_scr.step(scramble_action)
                
            rev_action = new_env_scr.rev_action(scramble_action)
                
            tempf_state, reward, scr_done , _ = new_env_scr.step(rev_action)
            
            #print('scrambler',np.concatenate((temps_state,fin_goal)), rev_action, reward, np.concatenate((tempf_state,fin_goal)), scr_done)
            agent.add_to_buffer(np.concatenate((temps_state,fin_goal)), rev_action, reward, np.concatenate((tempf_state,fin_goal)), scr_done)
            
            scr_state = temps_state
            
            if scr_done:
                pass
            
            if temp_done:
                pass

            if i%10 != 1:
                pass
            
            #Repeater Part
            rep_state = scr_state
            
            
            new_env_rep.init_state(rep_state)
            #print('rep_state',rep_state)
            
            for _ in range(5):
                
                #Choosing an action
                rep_action = agent.act(np.concatenate((rep_state,fin_goal)), eps) 
                
                #Executing that action
                rep_next_state, rep_reward, rep_done, _ = new_env_rep.step(rep_action)

                #Adding each data point to replay buffer    
                #print('repeater',np.concatenate((rep_state,fin_goal)), rep_action, rep_reward, np.concatenate((rep_next_state,fin_goal)), rep_done)
                agent.add_to_buffer(np.concatenate((rep_state,fin_goal)), rep_action, rep_reward, np.concatenate((rep_next_state,fin_goal)), rep_done)
                
                if rep_done:
                    break
        
        psuedo_goal = scr_state
        
        # Adding Foresight Goals  
        if not scr_done:
            flag = 0
            #print('foresight goal',psuedo_goal)
            
            #If not in hindsight_goals, then add to it
            if len(foresight_goals) == 0:
                foresight_goals.append(psuedo_goal)
                
            for fore_goal in foresight_goals:
                if ((psuedo_goal == fore_goal).all()):
                    flag = 1
                    
            if flag == 0:
                foresight_goals.append(psuedo_goal)
                
        
        # Learning from Foresight and Hindsight goals
        #print(len(hindsight_goals))
        
         ##MODULE HINDSIGHT LEARNING

        if i_episode % 25 == 0:
            hindsight_enable = True
            
        if hindsight_enable:
            for _ in range(min(3,len(hindsight_goals))):
        
            # Sample a hindsight goal
                hind_goal = random.choice(hindsight_goals)
            #print('hindgoal',hind_goal)
            
                for sublist in traj_val:
                
                    reward = sublist[2]
                    #print('reward',reward)
                #Altering the input state structure
                    new_state = np.concatenate((sublist[0],hind_goal))
                
                #Altering the reward
                    if ((sublist[3] == hind_goal).all() and reward <0):
                    #print('sublist[3]',sublist[3])
                    #print('hind_goal_in_check',hind_goal)
                    #print('entering check',(sublist[3] == hind_goal).all())
                        reward = 50
                    
                #print('updated reward',reward)
                    
                #Altering the next state structure
                    new_next_state = np.concatenate((sublist[3],hind_goal))
                #print('hindsight',new_state, sublist[1], reward, new_next_state, sublist[4])
                    agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])

            hindsight_enable = False
                
        ##MODULE FORESIGHT LEARNING
        if i_episode % 25 == 0:
            foresight_enable = True

        if foresight_enable:

            for _ in range(min(5,len(foresight_goals))):
            
                # Sample a hindsight goal
                fore_goal = random.choice(foresight_goals)
                #print('hindgoal',hind_goal)
                
                for sublist in traj_val:
                    
                    reward = sublist[2]
                    #print('reward',reward)
                    #Altering the input state structure
                    new_state = np.concatenate((sublist[0],fore_goal))
                    
                    #Altering the reward
                    if ((sublist[3] == fore_goal).all() and reward < 0):
                        reward = 100
                        
                    #print('updated reward',reward)
                        
                    #Altering the next state structure
                    new_next_state = np.concatenate((sublist[3],fore_goal))
                    #print('foresight',new_state, sublist[1], reward, new_next_state, sublist[4])
                    agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4]) 

            foresight_enable = False

                
                
        

        #Training (Refer to HER algorithm)
        for _ in range(max_t):
            agent.train_call()

        scores.append(score)
        scores_window.append(score)                       # save most recent score
        scores_window_printing.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps)                 # decrease epsilon
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score), end="")        
        if i_episode % 100 == 0: 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            scores.append(np.mean(scores_window))
        if np.mean(scores_window)>= reward_target:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
    return [np.array(scores),i_episode-100]

# n = 10 chain

env = NChainEnv(n=10)
new_env_scr = NChainEnv(n=10)
new_env_rep = NChainEnv(n=10)

state = env.reset()

state_shape = 1
action_shape = env.action_space.n

fin_goal = env.goal_state
new_env_scr.init_state(fin_goal)


agent = Agent_DQNHER(state_size=state_shape,action_size = action_shape,seed = 0)
scores_sre, terminal_ep_sre = dqn_sre(reward_target= 800,n_episodes= 10000, max_t= 100, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)


with open('sre_10chain.npy', 'wb') as f:
    np.save(f, scores_sre)


# n = 15 chain

env = NChainEnv(n=15)
new_env_scr = NChainEnv(n=15)
new_env_rep = NChainEnv(n=15)

state = env.reset()

state_shape = 1
action_shape = env.action_space.n

fin_goal = env.goal_state
new_env_scr.init_state(fin_goal)


agent = Agent_DQNHER(state_size=state_shape,action_size = action_shape,seed = 0)
scores_sre, terminal_ep_sre = dqn_sre(reward_target= 700,n_episodes= 10000, max_t= 100, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)


with open('sre_15chain.npy', 'wb') as f:
    np.save(f, scores_sre)

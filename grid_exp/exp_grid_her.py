# Imports

import numpy as np
import random

import torch
import numpy as np
from collections import deque, namedtuple

#import learning_agents
#from cube_env import Cube,visualise

import torch
import torch.nn as nn  
import torch.nn.functional as F

import numpy as np
import random

import torch
import numpy as np
from collections import deque, namedtuple

import Learning_Agents
import torch.optim as optim


import torch
import torch.nn as nn  
import torch.nn.functional as F

from grid_env import grid_nxn

# Building SRE-Agent (Complete Updated)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#n=5 grid
n = 5
grid = grid_nxn(n)

state_shape = grid.returnState().shape[0]
action_shape = 4

def dqn_her(n_episodes=1000, max_t=3*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995):
    print(pow(eps_decay,n_episodes))

    scores = []                 # list containing scores from each episode
    scores_window_printing = deque(maxlen=10) # For printing in the graph
    scores_window = deque(maxlen=100)  # last 100 scores for checking if the avg is more than 195
    eps = eps_start                    # initialize epsilon
    
    candidate_goals = deque(maxlen=10)
    
    
    env = grid_nxn(n)
    
    # Obtaining the start and end state for Tau and Tau_inv
    start_tau = env.returnState()
    goal_tau = env.returnGoalState()

    #Check if agent learns to solve a cube that is one move away from goal state
    for i_episode in range(1, n_episodes+1):
        
        state = env.reset()
        fin_goal = env.returnGoalState()
        score_main = 0
        done = False
        
        traj_val = []
         
        # Run for one trajectory (Original Task for evaluation)
        for t in range(max_t):
            
            #We store all we need for each trajectory
            
            #Choosing an action
            action = agent.act(np.concatenate((state,fin_goal)), eps)

            #Executing that action
            #next_state, reward, done, _ = env.step(action) (SLIGHTLY DIFFERENT FOR GRID WORLD)
            #Executing that action
            env.move(action)
            
            #Next state
            next_state = env.returnState()
            
            #Modified reward system
            reward = env.checkReward()
            
            #Checking if the episode ended
            if env.checkDone():
                done = True
             
            #print(state)
            traj_val.append([state,action,reward,next_state,done])
            
            
            #print('traj_val',traj_val)
            #print('len',len(traj_val))
            #agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score_main += reward
            
            if done:
                break 
                
        scores.append(score_main)  # save most recent score (For Eval purposes)
        
        # Hindsight relabelling of this trajectory:
        final_state = next_state
        
        # Hindsight relabelling for the explorer run
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]
            

            if (sublist[3] == final_state).all():
                reward = 1
                sublist[4] = True
                    
            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
            
                
        # Adding Hindsight Goals if final state is not in candidates:
        candidate = next_state
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(candidate_goals) == 0:
                candidate_goals.append(candidate)
                
            for hind_goal in candidate_goals:
                if ((candidate == hind_goal).all()):
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
        #print(len(traj_val))
        
        ####################### Running EXPLORER Module: ##############################
        # ITR Phase: Choosing a task_itr_goal (Task - tau)
        if len(candidate_goals) == 0:
            itr_goal_exp = goal_tau
        else:
            itr_goal_exp =  random.choices(candidate_goals)[0]
            
        done = False
        # Running the explorer run with itr_goal_exp as the desired goal state
        state = env.reset()
        traj_val = []
        reward = 0
        score = 0
        
        
        for _ in range(max_t):
            #We store all we need for each trajectory
            #Choosing an action
            action = agent.act(np.concatenate((state,itr_goal_exp)), eps)
            #Executing that action
            #next_state, reward, done, _ = env.step(action) (SLIGHTLY DIFFERENT FOR GRID WORLD)
            #Executing that action
            env.move(action)
            #Next state
            next_state = env.returnState()
            #Modified reward system
            if (itr_goal_exp == next_state).all():
                done = True
                reward = 1
            
            
            
            #Checking if the episode ended
            #if grid.checkDone():
            #    done = True
             
            #print(state)
            traj_val.append([state,action,reward,next_state,done])
            
            
            #print('traj_val',traj_val)
            #print('len',len(traj_val))
            #agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break 
                
        # Adding Hindsight Goals if final state is not in candidates:
        candidate = next_state
        
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(candidate_goals) == 0:
                candidate_goals.append(candidate)
                
            for hind_goal in candidate_goals:
                if ((candidate == hind_goal).all()):
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
        
        
        
        # Hindsight relabelling for the explorer run
        final_state = next_state
        
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]

            if (sublist[3] == final_state).all():
                reward = 1
                sublist[4] = True

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
    
        #Training (Refer to HER algorithm)
        for _ in range(max_t):
            agent.train_call()

        
        scores_window.append(score_main)             # save most recent score
        eps = max(eps_end, eps_decay*eps)                 # decrease epsilon
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score_main), end="")        
        if i_episode % 100 == 0: 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    return [np.array(scores),i_episode-100]


######### Running the experiments ##########::::::  

#n=5 grid
n = 15
grid = grid_nxn(n)

state_shape = grid.returnState().shape[0]
action_shape = 4

# seed = 0
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 0)
scores_dqn_1, terminal_ep_dqn_1 = dqn_her(n_episodes=10000, max_t=3*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

np.save('/mnt/data5/vihaan/grid_results/exp_grid_her_n15.npy', scores_dqn_1)

'''
# seed = 1
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 1)
scores_dqn_2, terminal_ep_dqn_2 = dqn_her(n_episodes=10000, max_t=3*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

# seed = 2
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 2)
scores_dqn_3, terminal_ep_dqn_3 = dqn_her(n_episodes=10000, max_t=3*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

avg_performance = (scores_dqn_1 + scores_dqn_2 + scores_dqn_3)/3

np.save('/mnt/data5/vihaan/grid_results/exp_grid_her_n5.npy', scores_dqn_1)

#n=10 grid
n = 10
grid = grid_nxn(n)

state_shape = grid.returnState().shape[0]
action_shape = 4

# seed = 0
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 0)
scores_dqn_1, terminal_ep_dqn_1 = dqn_her(n_episodes=10000, max_t=3*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

# seed = 1
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 1)
scores_dqn_2, terminal_ep_dqn_2 = dqn_her(n_episodes=10000, max_t=3*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

# seed = 2
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 2)
scores_dqn_3, terminal_ep_dqn_3 = dqn_her(n_episodes=10000, max_t=3*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

avg_performance = (scores_dqn_1 + scores_dqn_2 + scores_dqn_3)/3


np.save('/mnt/data5/vihaan/grid_results/exp_grid_her_n10.npy', scores_dqn_1)

#n=3 grid
n = 3
grid = grid_nxn(n)

state_shape = grid.returnState().shape[0]
action_shape = 4

# seed = 0
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 0)
scores_dqn_1, terminal_ep_dqn_1 = dqn_her(n_episodes=10000, max_t=3*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

# seed = 1
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 1)
scores_dqn_2, terminal_ep_dqn_2 = dqn_her(n_episodes=100000, max_t=3*n, eps_start=1.0, eps_end=0.1, eps_decay=0.99995)

# seed = 2
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 2)
scores_dqn_3, terminal_ep_dqn_3 = dqn_her(n_episodes=100000, max_t=3*n, eps_start=1.0, eps_end=0.1, eps_decay=0.99995)

avg_performance = (scores_dqn_1 + scores_dqn_2 + scores_dqn_3)/3

np.save('/mnt/data5/vihaan/grid_results/exp_grid_her_n3.npy', scores_dqn_1)
'''
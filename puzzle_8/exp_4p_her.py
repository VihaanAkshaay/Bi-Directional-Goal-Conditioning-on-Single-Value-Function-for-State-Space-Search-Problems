# Import all environments and SRE agent

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

#Importing Environment
from puzzle_env import EightPuzzle, FourPuzzle

############ 4Puzzle Experiments ############

env = FourPuzzle()
n = 4
new_env_rep = FourPuzzle()
new_env_scr = FourPuzzle()

state = env.reset()

state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

fin_goal = env.goal_state.flatten()

# Building SRE-NITR (Complete Updated)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def dqn_her(n_episodes=1000, max_t=8*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995):
    print(pow(eps_decay,n_episodes))

    scores = []                 # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores for checking if the avg is more than 195
    eps = eps_start                    # initialize epsilon
    
    candidate_goals = deque(maxlen=20)
    
    # Obtaining the start and end state for Tau and Tau_inv
    start_tau = env.get_state()
    goal_tau = env.goal_state.flatten()

    #Starting the whole run :)
    for i_episode in range(1, n_episodes+1):
        
        state = env.reset()
        fin_goal = env.goal_state.flatten()
        score_main = 0
        done = False
        
        traj_val = []
         
        # Run for one trajectory (Original Task for evaluation)
        for t in range(max_t):
            
            #We store all we need for each trajectory
            
            #Choosing an action
            action = agent.act(np.concatenate((state,fin_goal)), eps)

            #Executing that action
            next_state, reward, done, _ = env.step(action)

            traj_val.append([state,action,reward,next_state,done])
            
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
                if (candidate == hind_goal).all():
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
        
        
        for t in range(max_t):
            
            #Choosing an action
            action = agent.act(np.concatenate((state,itr_goal_exp)), eps)
            #Executing that action
            next_state, reward, done, _ = env.step(action) 
            
            traj_val.append([state,action,reward,next_state,done])
            
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
                if (candidate == hind_goal).all():
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


# seed = 0
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 0)
scores_dqn_1, terminal_ep_dqn_1 = dqn_her(n_episodes=10000, max_t=8*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

'''
# seed = 1
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 1)
scores_dqn_2, terminal_ep_dqn_2 = dqn_her(n_episodes=10000, max_t=8*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

# seed = 2
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 2)
scores_dqn_3, terminal_ep_dqn_3 = dqn_her(n_episodes=10000, max_t=8*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

avg_performance = (scores_dqn_1 + scores_dqn_2 + scores_dqn_3)/3
'''

np.save('/mnt/data5/vihaan/puzzle8_results/exp_4p_her.npy', scores_dqn_1)

############ 8Puzzle Experiments ############

env = EightPuzzle()
n = 8
new_env_rep = EightPuzzle()
new_env_scr = EightPuzzle()

state = env.reset()

state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

fin_goal = env.goal_state.flatten()

# Building SRE-NITR (Complete Updated)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def dqn_her(n_episodes=1000, max_t=8*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995):
    print(pow(eps_decay,n_episodes))

    scores = []                 # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores for checking if the avg is more than 195
    eps = eps_start                    # initialize epsilon
    
    candidate_goals = deque(maxlen=20)
    
    # Obtaining the start and end state for Tau and Tau_inv
    start_tau = env.get_state()
    goal_tau = env.goal_state.flatten()

    #Starting the whole run :)
    for i_episode in range(1, n_episodes+1):
        
        state = env.reset()
        fin_goal = env.goal_state.flatten()
        score_main = 0
        done = False
        
        traj_val = []
         
        # Run for one trajectory (Original Task for evaluation)
        for t in range(max_t):
            
            #We store all we need for each trajectory
            
            #Choosing an action
            action = agent.act(np.concatenate((state,fin_goal)), eps)

            #Executing that action
            next_state, reward, done, _ = env.step(action)

            traj_val.append([state,action,reward,next_state,done])
            
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
                if (candidate == hind_goal).all():
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
        
        
        for t in range(max_t):
            
            #Choosing an action
            action = agent.act(np.concatenate((state,itr_goal_exp)), eps)
            #Executing that action
            next_state, reward, done, _ = env.step(action) 
            
            traj_val.append([state,action,reward,next_state,done])
            
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
                if (candidate == hind_goal).all():
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


# seed = 0
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 0)
scores_dqn_1, terminal_ep_dqn_1 = dqn_her(n_episodes=10000, max_t=8*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

'''
# seed = 1
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 1)
scores_dqn_2, terminal_ep_dqn_2 = dqn_her(n_episodes=10000, max_t=8*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

# seed = 2
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 2)
scores_dqn_3, terminal_ep_dqn_3 = dqn_her(n_episodes=10000, max_t=8*n, eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

avg_performance = (scores_dqn_1 + scores_dqn_2 + scores_dqn_3)/3
'''

np.save('/mnt/data5/vihaan/puzzle8_results/exp_8p_her.npy', scores_dqn_1)

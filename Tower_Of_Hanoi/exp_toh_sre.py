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

from Tower_Env import HanoiEnv

############ 3 Disk Experiments ############

env = HanoiEnv()
new_env_rep = HanoiEnv()
new_env_scr = HanoiEnv()

num_disks = 3
env_noise = 0.0


env.set_env_parameters(num_disks, env_noise, verbose=False)
new_env_scr.set_env_parameters(num_disks, env_noise, verbose=False)
new_env_rep.set_env_parameters(num_disks, env_noise, verbose=False)

state = env.reset()

state_shape = num_disks
action_shape = env.action_space.n

fin_goal = env.goal_state
#print(fin_goal)
start_state = env.get_state()

# Building SRE-NITR (Complete Updated)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def dqn_sre(n_episodes=1000, max_t=3*pow(2,num_disks), eps_start=1.0, eps_end=0.01, eps_decay=0.9999):

    scores = []                 # list containing scores from each episode
    scores_window_printing = deque(maxlen=10) # For printing in the graph
    scores_window = deque(maxlen=100)  # last 100 scores for checking if the avg is more than 195
    eps = eps_start                    # initialize epsilon
    
    candidate_goals = deque(maxlen=20)
    candidate_itrs_tau = deque(maxlen=20)
    candidate_itrs_invtau = deque(maxlen=20)
    
    # Resetting all environments
    env.reset()
    new_env_scr.reset()
    new_env_rep.reset()
    
    # Obtaining the start and end state for Tau and Tau_inv
    start_tau = env.get_state()
    goal_tau = env.goal_state
    start_tau_inv = env.goal_state
    goal_tau_inv = env.get_state()




    #Starting the whole run :)
    for i_episode in range(1, n_episodes+1):
        
        state = env.reset()
        fin_goal = env.goal_state
        score_main = 0
        done = False
        
        traj_val = []

        # Run for one trajectory (Original Task for evaluation)
        #print('original task')
        #print('goal of original task', fin_goal)
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
        #print(traj_val)
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]
            

            if (sublist[3] == final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break
                    
            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
            
        # Adding Hindsight Goals if final state is not in candidates:
        candidate = next_state
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(candidate_goals) == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
        #print(len(traj_val))
        #print('candidate goals:',candidate_goals)
        
        ####################### Running EXPLORER Module: ##############################
        #print('starting explorer module')
        # ITR Phase: Choosing a task_itr_goal (Task - tau)
        if len(candidate_goals) == 0:
            itr_goal_exp = goal_tau
        else:
            itr_goal_exp =  random.choices(candidate_goals, candidate_itrs_tau)[0]
        #print('explorer goal',itr_goal_exp)
            
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
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
        
        #print(traj_val)
        
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
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
        
        #############################################################
        ###################### SCRAMBLER MODULE #####################
        #############################################################

        ############ Scrambler run with tau^star_inv as task ###########################
        #print('scrambler starting with star inv task')
        itr_goal_scr = goal_tau_inv
        #print('scrambler goal:',itr_goal_scr)
              
        new_env_scr.init_state(start_tau_inv)
        state = new_env_scr.get_state()
        reward = 0
        done = False
        traj_val = []
        #print('fin_goal',fin_goal)
        
        for i in range(max_t):
            
            #Scrambler Part
            action = agent.scr_act(np.concatenate((state,itr_goal_scr)), eps)
            
            #Executing that action
            next_state, reward, done, _ = new_env_scr.step(action) 
                
            #tempf_state, reward, scr_done , _ = new_env_scr.step(rev_action)
            traj_val.append([state,action,reward,next_state,done])

            if done:
                pass
            
            state = next_state
            
        # Hindsight relabelling for the scrambler run
        final_state = next_state
        
        #print(traj_val)
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]

            if (sublist[3] == final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])

            
        # Adding Hindsight Goals if final state is not in candidates:
        candidate = next_state
        
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(candidate_goals) == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))

        
        ############ Scrambler run with tau^star_inv-itr as task ###########################
        #print('scrambler with itr starting')
        
        #print()
        #Obtain goal for scrambler using itr_inv
        if len(candidate_goals) == 0:
            itr_goal_scr = goal_tau_inv
        else:
            itr_goal_scr =  random.choices(candidate_goals, candidate_itrs_invtau)[0]
        
        #print('scrambler goal',itr_goal_scr)
        #Start a new environment from the goal to generate samples
        new_env_scr.init_state(start_tau_inv)
        state = new_env_scr.get_state()
        reward = 0
        done = False
        traj_val = []
        #print('fin_goal',fin_goal)
        
        for i in range(max_t):
            
            #Scrambler Part
            action = agent.scr_act(np.concatenate((state,itr_goal_scr)), eps)
                
            #Executing that action
            next_state, reward, done, _ = new_env_scr.step(action) 
                
            #tempf_state, reward, scr_done , _ = new_env_scr.step(rev_action)
            traj_val.append([state,action,reward,next_state,done])

            if done:
                pass
            
            state = next_state
            
        #print(traj_val)
        ## TODO: Add this, the final state to the candidate list?
        # Adding Hindsight Goals if final state is not in candidates:
        candidate = next_state
        
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(candidate_goals) == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))

        # Scrambler hindsight relabelling:
        final_state = next_state
        
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]

            if (sublist[3]== final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
  
        ####################### REPEATER MODULE #######################
        #print('repeater module starts')
        #We start the repeater from the last state of the scrambler
        rep_state = next_state
        done = False
        reward = 0
        traj_val = []

        #print('repeater state:',rep_state)
        #print('repeater goal:', goal_tau)
        new_env_rep.init_state(rep_state)
            #print('rep_state',rep_state)
        state = rep_state
            
        for _ in range(max_t):
                
            #Choosing an action
            action = agent.scr_act(np.concatenate((state,goal_tau)), eps) 

            #Executing that action
            next_state, reward, done, _ = new_env_scr.step(action) 

            traj_val.append([state,action,reward,next_state,done])
            
            if done:
                break
                
            state = next_state
                
        # Hindsight relabelling for repeater trajectories
        final_state = next_state
        
        #print(traj_val)
        
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]

            if (sublist[3] == final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
        
    
        #Training (Refer to HER algorithm)
        for _ in range(max_t):
            agent.train_call()
            
        # Re evaluating itr values after training the agent:
        
        candidate_itrs_tau = deque(maxlen=20)
        candidate_itrs_invtau = deque(maxlen=20)
        
        for c in candidate_goals:
            candidate_itrs_tau.append(agent.itr_priority(start_tau,c,goal_tau))
            candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,c,goal_tau_inv))

        
        scores_window.append(score_main)       
        eps = max(eps_end, eps_decay*eps) 
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score_main), end="")        
        if i_episode % 100 == 0: 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    return [np.array(scores),i_episode-100]
# seed = 0
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 0)
scores_dqn_1, terminal_ep_dqn_1 = dqn_sre(n_episodes=10000, max_t=10*pow(2,num_disks), eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

#avg_performance = (scores_dqn_1 + scores_dqn_2 + scores_dqn_3)/3

np.save('/mnt/data5/vihaan/toh_results/exp_toh_sre_disc3.npy', scores_dqn_1)


############ 4 Disk Experiments ############

env = HanoiEnv()
new_env_rep = HanoiEnv()
new_env_scr = HanoiEnv()

num_disks = 4
env_noise = 0.0


env.set_env_parameters(num_disks, env_noise, verbose=False)
new_env_scr.set_env_parameters(num_disks, env_noise, verbose=False)
new_env_rep.set_env_parameters(num_disks, env_noise, verbose=False)

state = env.reset()

state_shape = num_disks
action_shape = env.action_space.n

fin_goal = env.goal_state
#print(fin_goal)
start_state = env.get_state()

# Building SRE-NITR (Complete Updated)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def dqn_sre(n_episodes=1000, max_t=3*pow(2,num_disks), eps_start=1.0, eps_end=0.01, eps_decay=0.9999):

    scores = []                 # list containing scores from each episode
    scores_window_printing = deque(maxlen=10) # For printing in the graph
    scores_window = deque(maxlen=100)  # last 100 scores for checking if the avg is more than 195
    eps = eps_start                    # initialize epsilon
    
    candidate_goals = deque(maxlen=20)
    candidate_itrs_tau = deque(maxlen=20)
    candidate_itrs_invtau = deque(maxlen=20)
    
    # Resetting all environments
    env.reset()
    new_env_scr.reset()
    new_env_rep.reset()
    
    # Obtaining the start and end state for Tau and Tau_inv
    start_tau = env.get_state()
    goal_tau = env.goal_state
    start_tau_inv = env.goal_state
    goal_tau_inv = env.get_state()




    #Starting the whole run :)
    for i_episode in range(1, n_episodes+1):
        
        state = env.reset()
        fin_goal = env.goal_state
        score_main = 0
        done = False
        
        traj_val = []

        # Run for one trajectory (Original Task for evaluation)
        #print('original task')
        #print('goal of original task', fin_goal)
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
        #print(traj_val)
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]
            

            if (sublist[3] == final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break
                    
            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
            
        # Adding Hindsight Goals if final state is not in candidates:
        candidate = next_state
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(candidate_goals) == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
        #print(len(traj_val))
        #print('candidate goals:',candidate_goals)
        
        ####################### Running EXPLORER Module: ##############################
        #print('starting explorer module')
        # ITR Phase: Choosing a task_itr_goal (Task - tau)
        if len(candidate_goals) == 0:
            itr_goal_exp = goal_tau
        else:
            itr_goal_exp =  random.choices(candidate_goals, candidate_itrs_tau)[0]
        #print('explorer goal',itr_goal_exp)
            
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
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
        
        #print(traj_val)
        
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
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
        
        #############################################################
        ###################### SCRAMBLER MODULE #####################
        #############################################################

        ############ Scrambler run with tau^star_inv as task ###########################
        #print('scrambler starting with star inv task')
        itr_goal_scr = goal_tau_inv
        #print('scrambler goal:',itr_goal_scr)
              
        new_env_scr.init_state(start_tau_inv)
        state = new_env_scr.get_state()
        reward = 0
        done = False
        traj_val = []
        #print('fin_goal',fin_goal)
        
        for i in range(max_t):
            
            #Scrambler Part
            action = agent.scr_act(np.concatenate((state,itr_goal_scr)), eps)
            
            #Executing that action
            next_state, reward, done, _ = new_env_scr.step(action) 
                
            #tempf_state, reward, scr_done , _ = new_env_scr.step(rev_action)
            traj_val.append([state,action,reward,next_state,done])

            if done:
                pass
            
            state = next_state
            
        # Hindsight relabelling for the scrambler run
        final_state = next_state
        
        #print(traj_val)
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]

            if (sublist[3] == final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])

            
        # Adding Hindsight Goals if final state is not in candidates:
        candidate = next_state
        
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(candidate_goals) == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))

        
        ############ Scrambler run with tau^star_inv-itr as task ###########################
        #print('scrambler with itr starting')
        
        #print()
        #Obtain goal for scrambler using itr_inv
        if len(candidate_goals) == 0:
            itr_goal_scr = goal_tau_inv
        else:
            itr_goal_scr =  random.choices(candidate_goals, candidate_itrs_invtau)[0]
        
        #print('scrambler goal',itr_goal_scr)
        #Start a new environment from the goal to generate samples
        new_env_scr.init_state(start_tau_inv)
        state = new_env_scr.get_state()
        reward = 0
        done = False
        traj_val = []
        #print('fin_goal',fin_goal)
        
        for i in range(max_t):
            
            #Scrambler Part
            action = agent.scr_act(np.concatenate((state,itr_goal_scr)), eps)
                
            #Executing that action
            next_state, reward, done, _ = new_env_scr.step(action) 
                
            #tempf_state, reward, scr_done , _ = new_env_scr.step(rev_action)
            traj_val.append([state,action,reward,next_state,done])

            if done:
                pass
            
            state = next_state
            
        #print(traj_val)
        ## TODO: Add this, the final state to the candidate list?
        # Adding Hindsight Goals if final state is not in candidates:
        candidate = next_state
        
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(candidate_goals) == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))

        # Scrambler hindsight relabelling:
        final_state = next_state
        
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]

            if (sublist[3]== final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
  
        ####################### REPEATER MODULE #######################
        #print('repeater module starts')
        #We start the repeater from the last state of the scrambler
        rep_state = next_state
        done = False
        reward = 0
        traj_val = []

        #print('repeater state:',rep_state)
        #print('repeater goal:', goal_tau)
        new_env_rep.init_state(rep_state)
            #print('rep_state',rep_state)
        state = rep_state
            
        for _ in range(max_t):
                
            #Choosing an action
            action = agent.scr_act(np.concatenate((state,goal_tau)), eps) 

            #Executing that action
            next_state, reward, done, _ = new_env_scr.step(action) 

            traj_val.append([state,action,reward,next_state,done])
            
            if done:
                break
                
            state = next_state
                
        # Hindsight relabelling for repeater trajectories
        final_state = next_state
        
        #print(traj_val)
        
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]

            if (sublist[3] == final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
        
    
        #Training (Refer to HER algorithm)
        for _ in range(max_t):
            agent.train_call()
            
        # Re evaluating itr values after training the agent:
        
        candidate_itrs_tau = deque(maxlen=20)
        candidate_itrs_invtau = deque(maxlen=20)
        
        for c in candidate_goals:
            candidate_itrs_tau.append(agent.itr_priority(start_tau,c,goal_tau))
            candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,c,goal_tau_inv))

        
        scores_window.append(score_main)       
        eps = max(eps_end, eps_decay*eps) 
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score_main), end="")        
        if i_episode % 100 == 0: 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    return [np.array(scores),i_episode-100]



# seed = 0
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 0)
scores_dqn_1, terminal_ep_dqn_1 = dqn_sre(n_episodes=10000, max_t=10*pow(2,num_disks), eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

np.save('/mnt/data5/vihaan/toh_results/exp_toh_sre_disc4.npy', scores_dqn_1)

############ 5 Disk Experiments ############

env = HanoiEnv()
new_env_rep = HanoiEnv()
new_env_scr = HanoiEnv()

num_disks = 5
env_noise = 0.0


env.set_env_parameters(num_disks, env_noise, verbose=False)
new_env_scr.set_env_parameters(num_disks, env_noise, verbose=False)
new_env_rep.set_env_parameters(num_disks, env_noise, verbose=False)

state = env.reset()

state_shape = num_disks
action_shape = env.action_space.n

fin_goal = env.goal_state
#print(fin_goal)
start_state = env.get_state()

# Building SRE-NITR (Complete Updated)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def dqn_sre(n_episodes=1000, max_t=3*pow(2,num_disks), eps_start=1.0, eps_end=0.01, eps_decay=0.9999):

    scores = []                 # list containing scores from each episode
    scores_window_printing = deque(maxlen=10) # For printing in the graph
    scores_window = deque(maxlen=100)  # last 100 scores for checking if the avg is more than 195
    eps = eps_start                    # initialize epsilon
    
    candidate_goals = deque(maxlen=20)
    candidate_itrs_tau = deque(maxlen=20)
    candidate_itrs_invtau = deque(maxlen=20)
    
    # Resetting all environments
    env.reset()
    new_env_scr.reset()
    new_env_rep.reset()
    
    # Obtaining the start and end state for Tau and Tau_inv
    start_tau = env.get_state()
    goal_tau = env.goal_state
    start_tau_inv = env.goal_state
    goal_tau_inv = env.get_state()




    #Starting the whole run :)
    for i_episode in range(1, n_episodes+1):
        
        state = env.reset()
        fin_goal = env.goal_state
        score_main = 0
        done = False
        
        traj_val = []

        # Run for one trajectory (Original Task for evaluation)
        #print('original task')
        #print('goal of original task', fin_goal)
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
        #print(traj_val)
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]
            

            if (sublist[3] == final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break
                    
            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
            
        # Adding Hindsight Goals if final state is not in candidates:
        candidate = next_state
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(candidate_goals) == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
        #print(len(traj_val))
        #print('candidate goals:',candidate_goals)
        
        ####################### Running EXPLORER Module: ##############################
        #print('starting explorer module')
        # ITR Phase: Choosing a task_itr_goal (Task - tau)
        if len(candidate_goals) == 0:
            itr_goal_exp = goal_tau
        else:
            itr_goal_exp =  random.choices(candidate_goals, candidate_itrs_tau)[0]
        #print('explorer goal',itr_goal_exp)
            
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
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
        
        #print(traj_val)
        
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
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
        
        #############################################################
        ###################### SCRAMBLER MODULE #####################
        #############################################################

        ############ Scrambler run with tau^star_inv as task ###########################
        #print('scrambler starting with star inv task')
        itr_goal_scr = goal_tau_inv
        #print('scrambler goal:',itr_goal_scr)
              
        new_env_scr.init_state(start_tau_inv)
        state = new_env_scr.get_state()
        reward = 0
        done = False
        traj_val = []
        #print('fin_goal',fin_goal)
        
        for i in range(max_t):
            
            #Scrambler Part
            action = agent.scr_act(np.concatenate((state,itr_goal_scr)), eps)
            
            #Executing that action
            next_state, reward, done, _ = new_env_scr.step(action) 
                
            #tempf_state, reward, scr_done , _ = new_env_scr.step(rev_action)
            traj_val.append([state,action,reward,next_state,done])

            if done:
                pass
            
            state = next_state
            
        # Hindsight relabelling for the scrambler run
        final_state = next_state
        
        #print(traj_val)
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]

            if (sublist[3] == final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])

            
        # Adding Hindsight Goals if final state is not in candidates:
        candidate = next_state
        
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(candidate_goals) == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))

        
        ############ Scrambler run with tau^star_inv-itr as task ###########################
        #print('scrambler with itr starting')
        
        #print()
        #Obtain goal for scrambler using itr_inv
        if len(candidate_goals) == 0:
            itr_goal_scr = goal_tau_inv
        else:
            itr_goal_scr =  random.choices(candidate_goals, candidate_itrs_invtau)[0]
        
        #print('scrambler goal',itr_goal_scr)
        #Start a new environment from the goal to generate samples
        new_env_scr.init_state(start_tau_inv)
        state = new_env_scr.get_state()
        reward = 0
        done = False
        traj_val = []
        #print('fin_goal',fin_goal)
        
        for i in range(max_t):
            
            #Scrambler Part
            action = agent.scr_act(np.concatenate((state,itr_goal_scr)), eps)
                
            #Executing that action
            next_state, reward, done, _ = new_env_scr.step(action) 
                
            #tempf_state, reward, scr_done , _ = new_env_scr.step(rev_action)
            traj_val.append([state,action,reward,next_state,done])

            if done:
                pass
            
            state = next_state
            
        #print(traj_val)
        ## TODO: Add this, the final state to the candidate list?
        # Adding Hindsight Goals if final state is not in candidates:
        candidate = next_state
        
        if not done:
            flag = 0
            
            #If not in hindsight_goals, then add to it
            if len(candidate_goals) == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))
                
            for hind_goal in candidate_goals:
                if (candidate == hind_goal).all():
                    flag = 1
                    
            if flag == 0:
                candidate_goals.append(candidate)
                candidate_itrs_tau.append(agent.itr_priority(start_tau,candidate,goal_tau))
                candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,candidate,goal_tau_inv))

        # Scrambler hindsight relabelling:
        final_state = next_state
        
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]

            if (sublist[3]== final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
  
        ####################### REPEATER MODULE #######################
        #print('repeater module starts')
        #We start the repeater from the last state of the scrambler
        rep_state = next_state
        done = False
        reward = 0
        traj_val = []

        #print('repeater state:',rep_state)
        #print('repeater goal:', goal_tau)
        new_env_rep.init_state(rep_state)
            #print('rep_state',rep_state)
        state = rep_state
            
        for _ in range(max_t):
                
            #Choosing an action
            action = agent.scr_act(np.concatenate((state,goal_tau)), eps) 

            #Executing that action
            next_state, reward, done, _ = new_env_scr.step(action) 

            traj_val.append([state,action,reward,next_state,done])
            
            if done:
                break
                
            state = next_state
                
        # Hindsight relabelling for repeater trajectories
        final_state = next_state
        
        #print(traj_val)
        
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],final_state))
            new_next_state = np.concatenate((sublist[3],final_state))
            #print('traj',new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            reward = sublist[2]

            if (sublist[3] == final_state).all():
                reward = 1
                sublist[4] = True
                #agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
                #break

            #print('adding',new_state, sublist[1], reward, new_next_state, sublist[4])
            agent.add_to_buffer(new_state, sublist[1], reward, new_next_state, sublist[4])
        
    
        #Training (Refer to HER algorithm)
        for _ in range(max_t):
            agent.train_call()
            
        # Re evaluating itr values after training the agent:
        
        candidate_itrs_tau = deque(maxlen=20)
        candidate_itrs_invtau = deque(maxlen=20)
        
        for c in candidate_goals:
            candidate_itrs_tau.append(agent.itr_priority(start_tau,c,goal_tau))
            candidate_itrs_invtau.append(agent.itr_priority(start_tau_inv,c,goal_tau_inv))

        
        scores_window.append(score_main)       
        eps = max(eps_end, eps_decay*eps) 
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score_main), end="")        
        if i_episode % 100 == 0: 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    return [np.array(scores),i_episode-100]


# seed = 0
agent = Learning_Agents.Agent_SREDQN(state_size=state_shape,action_size = action_shape,seed = 0)
scores_dqn_1, terminal_ep_dqn_1 = dqn_sre(n_episodes=10000, max_t=10*pow(2,num_disks), eps_start=1.0, eps_end=0.1, eps_decay=0.9995)

np.save('/mnt/data5/vihaan/toh_results/exp_toh_sre_disc5.npy', scores_dqn_1)






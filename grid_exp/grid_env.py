import numpy as np

class grid_nxn:
   
    def __init__(self,n):
        self.location = np.array([0,0])
        self.goal = np.array([n-1,n-1])
        self.n = n
        
    def returnGoalState(self):
        return self.goal
    #Dynamics
    # 0:up 1:right 2:down 3:left
    def move(self,action):
        if action == 0:
            self.location[0] += 1
            if self.location[0] == self.n:
                self.location[0] = self.n-1
            return self.location
        if action == 1:
            self.location[1] += 1
            if self.location[1] == self.n:
                self.location[1] = self.n-1
            return self.location
        if action == 2:
            self.location[0] -= 1
            if self.location[0] == -1:
                self.location[0] = 0
            return self.location
        if action == 3:
            self.location[1] -= 1
            if self.location[1] == -1:
                self.location[1] = 0
            return self.location
        
    def checkReward(self):
        if self.checkDone():
            return +1
        return 0
    
    def checkDone(self):
        if (self.location == self.goal).all():
            return True
        return False
        
    def returnState(self):
        return self.location
    
    def reset(self):
        self.location = np.array([0,0])
        return self.location
    
    def init_state(self,state_temp):
        self.location = state_temp
        return self.location
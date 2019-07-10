import os
import sys
import math
import numpy as np
class GridMDP:
    def __init__(self, grid, terminals, obstacles,actions, gamma, prob):
        self.terminals = terminals
        self.obstacles = obstacles
        self.actionlist = actions
        self.gamma = gamma
        self.reward = {}
        self.states = set()
        self.size = len(grid)
        self.prob=prob
        # self.MDP = MDP(actions, terminals, gamma)
        for x in range(len(grid)):#rows
            for y in range(len(grid)):
                self.reward[x,y] = grid[x][y]
                if(grid[x][y] != None):
                    self.states.add((x,y))
        #print(self.reward)
        #print(self.states)

    def R(self,state):
        return self.reward[state]

    def actions(self,state):
        #set of actions which can be performed.
        # print("in actions," , self.terminals, self.actionlist)
        if state in self.terminals:
            return [None]
        return self.actionlist
    def P(self, state, action):
        #P(s' |s,a)
        if(action == None):
            return [(state, 0.0)] #resultant state and probability of going there
        else:
            actionr=action+"R"
            actionl=action+"L"
            placesCanGo = []
            placesCanGo.append((self.go(state, actionr), (1-self.prob)/2))
            placesCanGo.append((self.go(state, actionl), (1-self.prob)/2))
            placesCanGo.append((self.go(state, action), self.prob))
            #print("possible places returning",placesCanGo)
            return placesCanGo
    def go(self, state, action):
        #return state that results if do action
        result = None
        if(action == 'R'):
            right = (0,1) #right i,j+1
            result = tuple(map(lambda x,y: x+y, right, state))
        elif(action == 'RR'):
            rightr = (1,1) #right i,j+1
            result = tuple(map(lambda x,y: x+y, rightr, state))
        elif(action == 'RL'):
            rightl = (-1,1) #right i,j+1
            result = tuple(map(lambda x,y: x+y, rightl, state))


        elif(action == 'L'):
            left = (0,-1) #left i,j-1
            result = tuple(map(lambda x,y: x+y, left, state))
        elif(action == 'LR'):
            leftr = (-1,-1) #right i,j+1
            result = tuple(map(lambda x,y: x+y, leftr, state))
        elif(action == 'LL'):
            leftl = (1,-1) #right i,j+1
            result = tuple(map(lambda x,y: x+y, leftl, state))


        elif(action == 'U'):
            up = (-1,0) #right i-1,j
            result = tuple(map(lambda x,y: x+y, up, state))
        elif(action == 'UR'):
            upr = (-1,1) #right i,j+1
            result = tuple(map(lambda x,y: x+y, upr, state))
        elif(action == 'UL'):
            upl = (-1,-1) #right i,j+1
            result = tuple(map(lambda x,y: x+y, upl, state))

        elif(action == 'D'):
            down = (1,0) #right i+1,j
            result = tuple(map(lambda x,y: x+y, down, state))
        elif(action == 'DR'):
            downr = (1,1) #right i,j+1
            result = tuple(map(lambda x,y: x+y, downr, state))
        elif(action == 'DL'):
            downl = (1,-1) #right i,j+1
            result = tuple(map(lambda x,y: x+y, downl, state))


        if(result[0] <0 or result[0]>=self.size or result[1]<0 or result[1]>=self.size or result in obstacles):
            return state #boundary states. 
        else:
            return result
def value_iteration(grid, epsilon):
    done = 1
    numberofIteration = 0
    U1 = dict([(s,0) for s in grid.states])
    gamma = grid.gamma
    reward = grid.reward

    while done :
        countstate = 0
        numberofIteration+=1
        U = U1.copy()
        delta = 0
        #print("old_iteration\n", U)
        #print("grid states",grid.states)
        for s in grid.states:
            # for a in grid.actions(s):
            # directionVal = sum([p * U[s1] for (s1, p) in grid.P(s, a)])
            #print("reward for state",s,grid.R(s))
            #print("grid actions",grid.actions(s) )
            U1[s] = grid.R(s) + gamma * max([sum([p * U[s1] for (s1, p) in grid.P(s, a)])
                                        for a in grid.actions(s)])
            #print("new util",U1[s])
            if(abs(U1[s] - U[s]) > delta):
                delta = abs(U1[s] - U[s])
            #print("delta", delta)
            if(delta < (epsilon *(1-gamma)/gamma)):
                countstate+=1
        if(delta < (epsilon *(1-gamma)/gamma) and countstate == len(grid.states)):
            done = 0;
    #print ("===", numberofIteration)
    return U
def expected_utility(grid, U, a,s):
    # print(a)
    return sum([p*U[s] for (s, p) in grid.P(s,a)])
def policy_matrix(grid, U):
    pi = {}
    for s in grid.states:
        max = float("-inf")
        maxAction = None
        for a in grid.actions(s):
            d = expected_utility(grid, U, a, s)
            if(max<d):
                max = d
                maxAction = a
        pi[s] = maxAction
    return pi
def turn_left(action):
    if(action == 'R'):
        return 'U'
    elif action == 'U':
        return 'L'
    elif action == 'L':
        return 'D'
    else:
        return 	'R'
def turn_right(action):
    if action == 'R':
        return 'D'
    elif action == 'D':
        return 'L'
    elif action == 'L':
        return 'U'
    else:
        return 'R'


fp = open("input.txt", "r")

s = int(fp.readline())#size of grid
o = int(fp.readline())#no of obstacles
grid = [-1] * s
for i in range(s):
    grid[i] = [-1] * s
    obstacles=[]
for i in range(o):
    obs = fp.readline().split(',')
    grid[int(obs[0])-1][int(obs[1])-1] = None
    obstacles.append((int(obs[0])-1,int(obs[1])-1))
endLocations = []
n = int(fp.readline())#no of terminal states
for i in range(n):
    endLocations.append(fp.readline());
actions = ['L', 'R', 'D', 'U']
terminals = []
for i in range(n):
    endlocs = endLocations[i].split(',')
    grid[int(endlocs[0])-1][int(endlocs[1])-1] = int(endlocs[2])
    #print("terminals", int(endlocs[0]),int(endlocs[1]))

    terminals.append((int(endlocs[0])-1,int(endlocs[1])-1))

    # grid = np.array(grid)
    # grid.transpose(1,0)
    # print(grid)
p=fp.readline()
p=float(p)
rp=fp.readline()
rp=float(rp)
gamma=fp.readline()
gamma=float(gamma)
for i in range(s):
    for j in range(s):
        if(grid[i][j]==-1):
            grid[i][j]=rp
#print(grid)
problem = GridMDP(grid, terminals, obstacles, actions, gamma, p)

utility = value_iteration(problem, 0.1)
#print("utility =====\n",utility)
utilitymatrix = [-1] * s
pi = policy_matrix(problem, utility)
#print(pi)
#ans=[0]*s
for i in range(s):
    for j in range(s):
        if((i,j) in obstacles):
            grid[i][j]='N'
        else:
            grid[i][j]=pi[(i,j)]
            if(grid[i][j]==None):
                grid[i][j]='E'
#print(grid)
target = open("output.txt", "w")
for i in range(s):
    for j in range(1,s):
        #print(ans[i][0])
        grid[i][0]+=","+grid[i][j]
    #print(grid[i][0])
    target.write(grid[i][0]+"\n")
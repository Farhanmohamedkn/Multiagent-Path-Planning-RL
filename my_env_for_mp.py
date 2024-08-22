import gym
from gym import spaces
import numpy as np
from collections import OrderedDict
from threading import Lock
import sys
from matplotlib.colors import hsv_to_rgb
# import colorsys
import random
import math
import copy
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
import torch.multiprocessing as mp
from multiprocessing import Lock, Array
import torch
# from gym.envs.classic_control import rendering  #in gym version gym version 0.21.0 it works
#      
manager = mp.Manager()
shared_dict_obs = manager.dict()
shared_dict_valid_act=manager.dict()
state_dict=manager.dict()



'''
    Observation: (position maps of current agent, current goal, other agents, other goals, obstacles)
        
    Action space: (Tuple)
        agent_id: positive integer
        action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST,
        5:NE, 6:SE, 7:SW, 8:NW}
    Reward: ACTION_COST for each action, GOAL_REWARD when robot arrives at target
'''
ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD,FINISH_REWARD,BLOCKING_COST = -0.3, -.5, 0.0, -2.,20.,-1.
opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
JOINT = False # True for joint estimation of rewards for closeby agents
dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
actionDict={v:k for k,v in dirDict.items()}



'''
State.
Implemented as 2 2d numpy arrays.
first one "state":
    static obstacle: -1
    empty: 0
    agent = positive integer (agent_id)
second one "goals":
    agent goal = positive int(agent_id)
'''
def init_state(world0, goals, diagonal, num_agents=1):

    
    assert(len(world0.shape) == 2 and world0.shape==goals.shape)
    state_dict['state'] = world0
    state_dict['goals'] = goals
    state_dict['num_agents'] = torch.tensor(num_agents).share_memory_()
    agents, agents_past, agent_goals = scanForAgents(num_agents, state_dict['state'],state_dict['goals'])
    state_dict['agents'] = agents
    state_dict['agents_past'] = agents_past
    state_dict['agent_goals'] = agent_goals
    state_dict['diagonal'] = torch.tensor(diagonal).share_memory_()
    print(state_dict['diagonal'])
    assert(len(agents) == num_agents)
    
    return state_dict 

def scanForAgents(num_agents, state, goals):
    agents = [(-1,-1) for i in range(num_agents)]
    agents_last = [(-1,-1) for i in range(num_agents)]        
    agent_goals = [(-1,-1) for i in range(num_agents)]        
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if(state[i,j]>0):
                agents[state[i,j]-1] = (i,j)
                agents_last[state[i,j]-1] = (i,j)
            if(goals[i,j]>0):
                agent_goals[goals[i,j]-1] = (i,j)
    assert((-1,-1) not in agents and (-1,-1) not in agent_goals)
    assert(agents==agents_last)
    agents= torch.tensor(agents).share_memory_()
    agents_last = torch.tensor(agents_last).share_memory_()
    agent_goals = torch.tensor(agent_goals).share_memory_()
    return agents, agents_last, agent_goals

def getPos(agent_id, agents):
   
    return agents[agent_id-1]

def getPastPos(agent_id, agents_past):
    
    return agents_past[agent_id-1]

def getGoal(agent_id, agent_goals):
    
    return agent_goals[agent_id-1]
    
def diagonalCollision(agent_id, newPos, agents, num_agents, agents_past):
    '''diagonalCollision(id,(x,y)) returns true if agent with id "id" collided diagonally with 
    any other agent in the state after moving to coordinates (x,y)
    agent_id: id of the desired agent to check for
    newPos: coord the agent is trying to move to (and checking for collisions)
    '''
#        def eq(f1,f2):return abs(f1-f2)<0.001
    def collide(a1,a2,b1,b2):
        '''
        a1,a2 are coords for agent 1, b1,b2 coords for agent 2, returns true if these collide diagonally
        '''
        return np.isclose( (a1[0]+a2[0]) /2. , (b1[0]+b2[0])/2. ) and np.isclose( (a1[1]+a2[1])/2. , (b1[1]+b2[1])/2. )
    assert(len(newPos) == 2)
    #up until now we haven't moved the agent, so getPos returns the "old" location
    lastPos = getPos(agent_id, agents)
    for agent in range(1,num_agents+1):
        if agent == agent_id: continue
        aPast = getPastPos(agent, agents_past)
        aPres = getPos(agent, agents)
        if collide(aPast,aPres,lastPos,newPos): return True
    return False

#try to move agent and return the status
def moveAgent(direction, agent_id, agents,agents_past, goals, state, diagonal):
    # with self.lock:
    ax=agents[agent_id-1][0]
    ay=agents[agent_id-1][1]

    # Not moving is always allowed
    if(direction==(0,0)):
        agents_past[agent_id-1]=agents[agent_id-1]
        return 1 if goals[ax,ay]==agent_id else 0

    # Otherwise, let's look at the validity of the move
    dx,dy =direction[0], direction[1]
    if(ax+dx>=state.shape[0] or ax+dx<0 or ay+dy>=state.shape[1] or ay+dy<0):#out of bounds
        return -1
    if(state[ax+dx,ay+dy]<0):#collide with static obstacle
        return -2
    if(state[ax+dx,ay+dy]>0):#collide with robot
        return -3
    # check for diagonal collisions
    if(diagonal):
        if diagonalCollision(agent_id,(ax+dx,ay+dy), agents,num_agents,agents_past):
            return -3
    # No collision: we can carry out the action
    state[ax,ay] = 0
    state[ax+dx,ay+dy] = agent_id 

    # Save the previous position of the agent
    agents_past[agent_id-1]=agents[agent_id-1]
    # print( agents[agent_id-1])
    # agents[agent_id-1] = (ax+dx,ay+dy)
    # Update the agent's position in the tensor
    #agent actually moved
    agents[agent_id-1, 0] = ax + dx  # Update x-coordinate
    agents[agent_id-1, 1] = ay + dy  # Update y-coordinate

    # Check if the agent has reached its goal
    if goals[ax+dx,ay+dy]==agent_id:
        return 1
    elif goals[ax+dx,ay+dy]!=agent_id and goals[ax,ay]==agent_id:
        return 2
    else:
        return 0

    # try to execture action and return whether action was executed or not and why
    #returns:
    #     2: action executed and left goal
    #     1: action executed and reached goal (or stayed on)
    #     0: action executed
    #    -1: out of bounds
    #    -2: collision with wall
    #    -3: collision with robot
def getDir(action):
    return dirDict[action]

def act(action, agent_id, agents,agents_past, goals, state, diagonal):
    # 0     1  2  3  4 
    # still N  E  S  W
    direction = getDir(action)
    # direction=torch.tensor(direction).share_memory_()
    moved = moveAgent(direction,agent_id, agents,agents_past, goals, state, diagonal)
    return moved


def getAction(direction):
    return actionDict[direction]

# Compare with a plan to determine job completion
def done(agents, goals):
    # with self.lock:
    numComplete = 0
    for i in range(1,len(agents)+1):
        agent_pos = agents[i-1]
        if goals[agent_pos[0],agent_pos[1]] == i:
            numComplete += 1
    return numComplete==len(agents) #, numComplete/float(len(self.agents))


class MAPFEnv(gym.Env):
    def getFinishReward(self):
        return FINISH_REWARD
    metadata = {"render.modes": ["human", "ansi"]}

    # Initialize env
    # print("game ENV initialized")
    def __init__(self, num_agents=1, observation_size=10,world0=None, goals0=None, DIAGONAL_MOVEMENT=False, SIZE=(10,40), PROB=(0,.5), FULL_HELP=False,blank_world=False):
        """
        Args:
            DIAGONAL_MOVEMENT: if the agents are allowed to move diagonally
            SIZE: size of a side of the square grid
            PROB: range of probabilities that a given block is an obstacle
            FULL_HELP
        """
        # Initialize member variables
        self.num_agents        = num_agents 
        #a way of doing joint rewards
        self.individual_rewards           = [0 for i in range(num_agents)]
        self.observation_size  = observation_size
        self.SIZE              = SIZE
        self.PROB              = PROB
        self.fresh             = True
        self.FULL_HELP         = FULL_HELP
        self.finished          = False
        self.DIAGONAL_MOVEMENT = DIAGONAL_MOVEMENT

        # Initialize data structures
        self._setWorld(world0,goals0,blank_world=blank_world)
        if DIAGONAL_MOVEMENT:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(9)])
        else:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(5)])
        self.viewer           = None

    def isConnected(self,world0):
        sys.setrecursionlimit(10000)
        world0 = world0.copy()

        def firstFree(world0):
            for x in range(world0.shape[0]):
                for y in range(world0.shape[1]):
                    if world0[x,y]==0:
                        return x,y
        def floodfill(world,i,j):
            sx,sy=world.shape[0],world.shape[1]
            if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                return
            if(world[i,j]==-1):return
            world[i,j] = -1
            floodfill(world,i+1,j)
            floodfill(world,i,j+1)
            floodfill(world,i-1,j)
            floodfill(world,i,j-1)

        i,j = firstFree(world0)
        floodfill(world0,i,j)
        if np.any(world0==0):
            return False
        else:
            return True

    def getObstacleMap(self):
        return (self.world['state']==-1).astype(int)
    
    def getGoals(self):
        result=[]
        for i in range(1,self.num_agents+1):
            result.append(getGoal(i, self.world['agent_goals']))
        return result
    
    def getPositions(self):
        result=[]
        for i in range(1,self.num_agents+1):
            result.append(getPos(i, self.world['agents']))
        return result
    
    def _setWorld(self, world0=None, goals0=None,blank_world=False):
       
        #blank_world is a flag indicating that the world given has no agent or goal positions 
        def getConnectedRegion(world,regions_dict,x,y):
            sys.setrecursionlimit(1000000)
            '''returns a list of tuples of connected squares to the given tile
            this is memoized with a dict'''
            if (x,y) in regions_dict:
                return regions_dict[(x,y)]
            visited=set()
            sx,sy=world.shape[0],world.shape[1]
            work_list=[(x,y)]
            while len(work_list)>0:
                (i,j)=work_list.pop()
                if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                    continue
                if(world[i,j]==-1):
                    continue#crashes
                if world[i,j]>0:
                    regions_dict[(i,j)]=visited
                if (i,j) in visited:continue
                visited.add((i,j))
                work_list.append((i+1,j))
                work_list.append((i,j+1))
                work_list.append((i-1,j))
                work_list.append((i,j-1))
            regions_dict[(x,y)]=visited
            return visited
        #defines the State object, which includes initializing goals and agents
        #sets the world to world0 and goals, or if they are None randomizes world
        if not (world0 is None):
            print("not going in this condition")
            if goals0 is None and not blank_world:
                print("not going in this condition")
                raise Exception("you gave a world with no goals!")
            if blank_world:
                #RANDOMIZE THE POSITIONS OF AGENTS
                print("not going in this condition")
                agent_counter = 1
                agent_locations=[]
                while agent_counter<=self.num_agents:
                    x,y       = np.random.randint(0,world0.shape[0]),np.random.randint(0,world0.shape[1])
                    if(world0[x,y] == 0):
                        world0[x,y]=agent_counter
                        agent_locations.append((x,y))
                        agent_counter += 1   
                #RANDOMIZE THE GOALS OF AGENTS
                goals0 = np.zeros(world0.shape).astype(int)
                goal_counter = 1
                agent_regions=dict()  
                while goal_counter<=self.num_agents:
                    agent_pos=agent_locations[goal_counter-1]
                    valid_tiles=getConnectedRegion(world0,agent_regions,agent_pos[0],agent_pos[1])#crashes
                    x,y  = random.choice(list(valid_tiles))
                    if(goals0[x,y]==0 and world0[x,y]!=-1):
                        goals0[x,y]    = goal_counter
                        goal_counter += 1
                self.initial_world = world0.copy()
                self.initial_goals = goals0.copy()
                self.world = init_state(self.initial_world,self.initial_goals,self.DIAGONAL_MOVEMENT,self.num_agents)
                print("here")
                return
            self.initial_world = world0
            self.initial_goals = goals0
            self.world = init_state(world0,goals0,self.DIAGONAL_MOVEMENT,self.num_agents)

            print("....................................inside sETWORL if condition.........................")
            return

        #otherwise we have to randomize the world
        print("below code is run always init and reset find out how agents go in mapf")
        #RANDOMIZE THE STATIC OBSTACLES
        prob=np.random.triangular(self.PROB[0],.33*self.PROB[0]+.66*self.PROB[1],self.PROB[1])
        size=np.random.choice([self.SIZE[0],self.SIZE[0]*.5+self.SIZE[1]*.5,self.SIZE[1]],p=[.5,.25,.25])
        world     = -(np.random.rand(int(size),int(size))<prob).astype(int)

        #RANDOMIZE THE POSITIONS OF AGENTS
        agent_counter = 1
        agent_locations=[]
        while agent_counter<=self.num_agents:
            x,y       = np.random.randint(0,world.shape[0]),np.random.randint(0,world.shape[1])
            if(world[x,y] == 0):
                world[x,y]=agent_counter
                agent_locations.append((x,y))
                agent_counter += 1  
        # print("start position of mutiple  agents we are training=",agent_locations)    # to print the starting pos of current agent we are considering #farhan  
        
        #RANDOMIZE THE GOALS OF AGENTS
        goals = np.zeros(world.shape).astype(int)
        goal_counter = 1
        agent_regions=dict()     
        while goal_counter<=self.num_agents:
            agent_pos=agent_locations[goal_counter-1]
            valid_tiles=getConnectedRegion(world,agent_regions,agent_pos[0],agent_pos[1])
            x,y  = random.choice(list(valid_tiles))
            if(goals[x,y]==0 and world[x,y]!=-1):
                goals[x,y]    = goal_counter
                goal_counter += 1
        self.initial_world = world
        self.initial_goals = goals
        world1=torch.tensor(world).share_memory_()
        goals1=torch.tensor(goals).share_memory_()

        # Convert numpy arrays to shared memory
        # world,goals = store_the_reseted_world_to_share(world,goals)
        
        #convert back to numpy array
        # world =world.numpy()
        # goals =goals.numpy()

        # Initialize the shared State class
        print("new world created")
        self.world = init_state(world1, goals1,self.DIAGONAL_MOVEMENT,num_agents=self.num_agents)
        # print(self.world, self.world['agents'])
       


    # Returns an observation of an agent
    def _observe(self,agent_id):
        assert(agent_id>0)
        top_left=(getPos(agent_id, self.world['agents'])[0]-self.observation_size//2,getPos(agent_id, self.world['agents'])[1]-self.observation_size//2)
        bottom_right=(top_left[0]+self.observation_size,top_left[1]+self.observation_size)        
        obs_shape=(self.observation_size,self.observation_size)
        goal_map             = np.zeros(obs_shape)
        poss_map             = np.zeros(obs_shape)
        goals_map            = np.zeros(obs_shape)
        obs_map              = np.zeros(obs_shape)
        visible_agents=[]
        for i in range(top_left[0],top_left[0]+self.observation_size):
            for j in range(top_left[1],top_left[1]+self.observation_size):
                if i>=self.world['state'].shape[0] or i<0 or j>=self.world['state'].shape[1] or j<0:
                    #out of bounds, just treat as an obstacle
                    obs_map[i-top_left[0],j-top_left[1]]=1
                    continue
                if self.world['state'][i,j]==-1:
                    #obstacles
                    obs_map[i-top_left[0],j-top_left[1]]=1
                if self.world['state'][i,j]==agent_id:
                    #agent's position
                    poss_map[i-top_left[0],j-top_left[1]]=1
                if self.world['goals'][i,j]==agent_id:
                    #agent's goal
                    goal_map[i-top_left[0],j-top_left[1]]=1
                if self.world['state'][i,j]>0 and self.world['state'][i,j]!=agent_id:
                    #other agents' positions
                    visible_agents.append(self.world['state'][i,j])
                    poss_map[i-top_left[0],j-top_left[1]]=1

        for agent in visible_agents:
            x, y = getGoal(agent, self.world['agent_goals'])
            min_node = (max(top_left[0], min(top_left[0] + self.observation_size - 1, x)),
                        max(top_left[1], min(top_left[1] + self.observation_size - 1, y)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx=getGoal(agent_id, self.world['agent_goals'])[0]-getPos(agent_id, self.world['agents'])[0]  
        dy=getGoal(agent_id, self.world['agent_goals'])[1]-getPos(agent_id, self.world['agents'])[1]
        mag=(dx**2+dy**2)**.5 # magnitude for finidng unit vector to the goal 
        if mag!=0:
            dx=dx/mag
            dy=dy/mag

        
        
        poss_map=torch.tensor(poss_map).share_memory_()
        goal_map=torch.tensor(goal_map).share_memory_()
        goals_map=torch.tensor(goals_map).share_memory_()
        obs_map=torch.tensor(obs_map).share_memory_()
        obs_map=obs_map.share_memory_()
        dx= dx.share_memory_()
        dy= dy.share_memory_()
        mag=mag.share_memory_()
        
        
        shared_dict_obs[agent_id] = [[poss_map,goal_map,goals_map,obs_map],[dx,dy,mag]]
        return shared_dict_obs




    # Resets environment
    def _reset(self, agent_id,world0=None,goals0=None):
        self.finished = False
        # self.mutex.acquire()
        
        # Initialize data structures
        self._setWorld(world0,goals0)
        self.fresh = True
        
        # self.mutex.release()
        if self.viewer is not None:
            self.viewer = None
        on_goal = getPos(agent_id, self.world['agents']) == getGoal(agent_id, self.world['agent_goals'])
        #we assume you don't start blocking anyone (the probability of this happening is insanely low)
        return self._listNextValidActions(agent_id), on_goal,False

    def _complete(self):
        return done(self.world['agents'], self.world['goals'])
    
    def getAstarCosts(self,start, goal):
        #returns a numpy array of same dims as self.world['state'] with the distance to the goal from each coord
        def lowestF(fScore,openSet):
            #find entry in openSet with lowest fScore
            assert(len(openSet)>0)
            minF=2**31-1
            minNode=None
            for (i,j) in openSet:
                if (i,j) not in fScore:continue
                if fScore[(i,j)]<minF:
                    minF=fScore[(i,j)]
                    minNode=(i,j)
            return minNode      
        def getNeighbors(node):
            #return set of neighbors to the given node
            n_moves=9 if self.DIAGONAL_MOVEMENT else 5
            neighbors=set()
            for move in range(1,n_moves):#we dont want to include 0 or it will include itself
                direction=getDir(move)
                dx=direction[0]
                dy=direction[1]
                ax=node[0]
                ay=node[1]
                if(ax+dx>=self.world['state'].shape[0] or ax+dx<0 or ay+dy>=self.world['state'].shape[1] or ay+dy<0):#out of bounds
                    continue
                if(self.world['state'][ax+dx,ay+dy]==-1):#collide with static obstacle
                    continue
                neighbors.add((ax+dx,ay+dy))
            return neighbors
        
        #NOTE THAT WE REVERSE THE DIRECTION OF SEARCH SO THAT THE GSCORE WILL BE DISTANCE TO GOAL
        start,goal=goal,start
        
        # The set of nodes already evaluated
        closedSet = set()
    
        # The set of currently discovered nodes that are not evaluated yet.
        # Initially, only the start node is known.
        openSet =set()
        openSet.add(start)
    
        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many nodes, cameFrom will eventually contain the
        # most efficient previous step.
        cameFrom = dict()
    
        # For each node, the cost of getting from the start node to that node.
        gScore =dict()#default value infinity
    
        # The cost of going from start to start is zero.
        gScore[start] = 0
    
        # For each node, the total cost of getting from the start node to the goal
        # by passing by that node. That value is partly known, partly heuristic.
        fScore = dict()#default infinity
    
        #our heuristic is euclidean distance to goal
        heuristic_cost_estimate = lambda x,y:math.hypot(x[0]-y[0],x[1]-y[1])
        
        # For the first node, that value is completely heuristic.        
        fScore[start] = heuristic_cost_estimate(start, goal)
    
        while len(openSet) != 0:
            #current = the node in openSet having the lowest fScore value
            current = lowestF(fScore,openSet)
    
            openSet.remove(current)
            closedSet.add(current)
            for neighbor in getNeighbors(current):
                if neighbor in closedSet:
                    continue		# Ignore the neighbor which is already evaluated.
            
                if neighbor not in openSet:	# Discover a new node
                    openSet.add(neighbor)
                
                # The distance from start to a neighbor
                #in our case the distance between is always 1
                tentative_gScore = gScore[current] + 1
                if tentative_gScore >= gScore.get(neighbor,2**31-1):
                    continue		# This is not a better path.
            
                # This path is the best until now. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + heuristic_cost_estimate(neighbor, goal) 
    
        #parse through the gScores
        costs=self.world['state'].copy()
        for (i,j) in gScore:
            costs[i,j]=gScore[i,j]
        return costs
    
    def astar(self,world,start,goal,robots=[]):
        '''robots is a list of robots to add to the world'''
        for (i,j) in robots:
            world[i,j]=1
        try:
            path=cpp_mstar.find_path(world,[start],[goal],1,5)
        except NoSolutionError:
            path=None
        for (i,j) in robots:
            world[i,j]=0
        return path
    
    def get_blocking_reward(self,agent_id):
        '''calculates how many robots the agent is preventing from reaching goal
        and returns the necessary penalty'''
        #accumulate visible robots
        other_robots=[]
        other_locations=[]
        inflation=10
        top_left=(getPos(agent_id, self.world['agents'])[0]-self.observation_size//2,getPos(agent_id, self.world['agents'])[1]-self.observation_size//2)
        bottom_right=(top_left[0]+self.observation_size,top_left[1]+self.observation_size)        
        for agent in range(1,self.num_agents):
            if agent==agent_id: continue
            x,y=getPos(agent, self.world['agents'])
            if x<top_left[0] or x>=bottom_right[0] or y>=bottom_right[1] or y<top_left[1]:
                continue
            other_robots.append(agent)
            other_locations.append((x,y))
        num_blocking=0
        world=self.getObstacleMap()
        for agent in other_robots:
            other_locations.remove(getPos(agent, self.world['agents']))
            #before removing
            path_before=self.astar(world,getPos(agent, self.world['agents']),getGoal(agent, self.world['agent_goals']),
                                   robots=other_locations+[getPos(agent_id, self.world['agents'])])
            #after removing
            path_after=self.astar(world,getPos(agent, self.world['agents']),getGoal(agent, self.world['agent_goals']),
                                   robots=other_locations)
            other_locations.append(getPos(agent, self.world['agents']))
            if (path_before is None and path_after is None):continue
            if (path_before is not None and path_after is None):continue
            if (path_before is None and path_after is not None)\
                or len(path_before)>len(path_after)+inflation:
                num_blocking+=1
        return num_blocking*BLOCKING_COST
            
        
    # Executes an action by an agent
    def _step(self, action_input,episode=0):
        #episode is an optional variable which will be used on the reward discounting
        self.fresh = False
        n_actions = 9 if self.DIAGONAL_MOVEMENT else 5

        # Check action input
        assert len(action_input) == 2, 'Action input should be a tuple with the form (agent_id, action)'
        assert action_input[1] in range(n_actions), 'Invalid action'
        assert action_input[0] in range(1, self.num_agents+1)

        # Parse action input
        agent_id = action_input[0]
        action   = action_input[1]

        

        #get start location of agent
        agentStartLocation = getPos(agent_id, self.world['agents'])
        
        # Execute action & determine reward
        action_status = act(action,agent_id, self.world['agents'], self.world['agents_past'], self.world['goals'], self.world['state'], self.world['diagonal'])
        valid_action=action_status >=0
        #     2: action executed and left goal
        #     1: action executed and reached/stayed on goal
        #     0: action executed
        #    -1: out of bounds
        #    -2: collision with wall
        #    -3: collision with robot
        blocking=False
        
        if action==0:#staying still
            if action_status == 1:#stayed on goal
                reward=GOAL_REWARD
                x=self.get_blocking_reward(agent_id)
                reward+=x
                if x<0:
                    blocking=True
            elif action_status == 0:#stayed off goal
                reward=IDLE_COST
        else:#moving
            if (action_status == 1): # reached goal
                reward = GOAL_REWARD
            elif (action_status == -3 or action_status==-2 or action_status==-1): # collision
                reward = COLLISION_REWARD
            elif (action_status == 2): #left goal
                reward=ACTION_COST
            else:
                reward=ACTION_COST
        self.individual_rewards[agent_id-1]=reward

        if JOINT:
            visible=[False for i in range(self.num_agents)]
            v=0
            #joint rewards based on proximity
            for agent in range(1,self.num_agents+1):
                #tally up the visible agents
                if agent==agent_id:
                    continue
                top_left=(getPos(agent_id, self.world['agents'])[0]-self.observation_size//2, \
                          getPos(agent_id, self.world['agents'])[1]-self.observation_size//2)
                pos=getPos(agent, self.world['agents'])
                if pos[0]>=top_left[0] and pos[0]<top_left[0]+self.observation_size\
                    and pos[1]>=top_left[1] and pos[1]<top_left[1]+self.observation_size:
                        #if the agent is within the bounds for observation
                        v+=1
                        visible[agent-1]=True
            if v>0:
                reward=self.individual_rewards[agent_id-1]/2
                #set the reward to the joint reward if we are 
                for i in range(self.num_agents):
                    if visible[i]:
                        reward+=self.individual_rewards[i]/(v*2)

        # Perform observation
        state = self._observe(agent_id) 

        # Done?
        doneVar = done(self.world['agents'], self.world['goals'])
        self.finished |= doneVar

        # next valid actions
        nextActions = self._listNextValidActions(agent_id, action,episode=episode)

        # on_goal estimation
        on_goal = getPos(agent_id, self.world['agents']) == getGoal(agent_id, self.world['agent_goals'])
        
    
        

        return state, reward, doneVar, nextActions, on_goal, blocking, valid_action,action_status

    def _listNextValidActions(self, agent_id, prev_action=0,episode=0):
        available_actions = [0] # staying still always allowed

        # Get current agent position
        agent_pos = getPos(agent_id, self.world['agents'])
        ax,ay     = agent_pos[0],agent_pos[1]
        n_moves   = 9 if self.DIAGONAL_MOVEMENT else 5

        for action in range(1,n_moves):
            direction = getDir(action)
            dx,dy     = direction[0],direction[1]
            if(ax+dx>=self.world['state'].shape[0] or ax+dx<0 or ay+dy>=self.world['state'].shape[1] or ay+dy<0):#out of bounds
                continue
            if(self.world['state'][ax+dx,ay+dy]<0):#collide with static obstacle
                continue
            if(self.world['state'][ax+dx,ay+dy]>0):#collide with robot
                continue
            # check for diagonal collisions
            if(self.DIAGONAL_MOVEMENT):
                if diagonalCollision(agent_id,(ax+dx,ay+dy), self.world['agents'], self.world['num_agents'], self.world['agents_past']):
                    continue          
            #otherwise we are ok to carry out the action
            available_actions.append(action)

        if opposite_actions[prev_action] in available_actions:
            available_actions.remove(opposite_actions[prev_action])
        
        shared_dict_valid_act[agent_id]=torch.tensor(available_actions).share_memory_()      
        return shared_dict_valid_act

    
def worker(agent_id,env,episode_count,lock,barrier):
    
    cur_pos=env.getPositions()
    goals=env.getGoals()
    # print("before step agent=",agent_id,cur_pos,goals)
    print("before step") 
    barrier.wait()
    with lock:
        observation  = env._observe(agent_id)
        validActions = env._listNextValidActions(agent_id)
        a=random.choice(validActions[agent_id].tolist())
    
    barrier.wait()
    print("for agent",agent_id,"=",observation)
    barrier.wait()
    with lock:
        state, reward, done, nextActions, on_goal, blocking, valid_action=env._step((agent_id,a))
    barrier.wait()
    print("after 1 step") 
    with lock:
        validActions = env._listNextValidActions(agent_id,a)
        a=random.choice(validActions[agent_id].tolist())
        observation=env._observe(agent_id)
    barrier.wait()
    print("for agent",agent_id,"=",observation)
    barrier.wait()
    # cur_pos=env.getPositions()
    # goals=env.getGoals()
    # print("after step agent=",agent_id,cur_pos,goals)
    if agent_id==1:
        env._reset(agent_id)
    
    barrier.wait()
    print("after reset")
    with lock:
        observation=env._observe(agent_id)
    barrier.wait()
    print("for agent",agent_id,"=",observation)
    

if __name__=='__main__':
    num_agents=2
    env=MAPFEnv(num_agents,PROB=(.3,.5),SIZE=(10,11),DIAGONAL_MOVEMENT=False)
    world=env.world
    episode_count=mp.Value('i', 0)
    lock = mp.Lock()
    barrier = mp.Barrier(num_agents)
    
    
    processes = []
    for agent_id in range(1,num_agents+1):
        print(f'worker {agent_id}: started')
        p = mp.Process(target=worker, args=(agent_id,env,episode_count,lock,barrier))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

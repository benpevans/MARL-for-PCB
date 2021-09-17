import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import clear_output

random.seed(a=54)

class Board:

    CELL_TYPES_DICT = {
        "empty" : 0,
        "wall"  : 10
    }

    ACTIONS_DICT = {
        "upleft"   : 0,
        "up"       : 1,
        "upright"  : 2,
        "left"     : 3,
        "right"    : 4,
        "downleft" : 5,
        "down"     : 6,
        "downright": 7
    }

    def __init__(
        self, 
        boardsize=(10,20),
        obstacles=0,
        obstacle_locations=None,
        paddingsize = None
        ):
        
        self.boardsize = boardsize
        self.board = np.zeros(boardsize, int)
        if obstacle_locations:
            for location in obstacle_locations:
                self.board[location] = self.CELL_TYPES_DICT["wall"]
        else:
            for i in range(obstacles):
                coord = self.find_empty()
                self.board[coord] = self.CELL_TYPES_DICT["wall"]
                # add obstacle isn't working
                # self.add_obstacle(coord)

        if paddingsize:
            self.padding = np.ones((self.boardsize[0]+paddingsize*2,\
                self.boardsize[1]+paddingsize*2))*self.CELL_TYPES_DICT["wall"]
            self.padding[paddingsize:self.boardsize[0]+paddingsize,\
                paddingsize:self.boardsize[1]+paddingsize] = self.board
            self.board = self.padding

    def change_cell(self, coord, value):
        self.board[coord] = value

    def get_partial_obs(self, path, partial_obs):

        # total obs size
        obs_size = 2*partial_obs + 1
        
        # find the center of the observation
        center = path.path_head()

        # check partial observation is within board boundaries
        l = center[0] - partial_obs
        r = center[0] + partial_obs
        d = center[1] - partial_obs
        u = center[1] + partial_obs

        # find the boundaries of the reference frame, 
        # l,r,d,u = coordinates of global board that are observable, 
        # lpad, rpad, upad, dpad = coordinates of padded board that contain observation, 
        # rpad - lpad = r - l, upad - dpad = u - d
        if l >= 0:
            lpad = 0
        else:
            lpad = -l
            l = 0
        if r < self.boardsize[0]:
            r += 1
            rpad = obs_size
        else:
            rpad = obs_size - (r-self.boardsize[0]) - 1
            r = self.boardsize[0]
        if d >= 0:
            dpad = 0
        else:
            dpad = -d
            d = 0
        if u < self.boardsize[1]:
            upad = obs_size
            u += 1
        else:
            upad = obs_size - (u-self.boardsize[1]) - 1
            u = self.boardsize[1]

        # create a partial observation of all walls
        obs = np.ones((obs_size, obs_size),int)*self.CELL_TYPES_DICT["wall"]

        # convert to the observation taken from the global state
        obs[lpad:rpad, dpad:upad] = self.board[l:r, d:u]

        return obs

    # def add_obstacle(self, coord):
    #     # this function doesn't work yet
    #     """
    #     Adds different obstacles to board with different shapes and various probabilities
    #     Input: Obstacle Coordinates
    #     Shapes:
    #     1. x
    #     2. xx
    #     3.  x
    #        xx
    #     """
    #     x, y = coord
    #     self.board[coord] = self.CELL_TYPES_DICT["wall"]
    #     rand = random.random()

    #     # there is no check on if the cell is empty or not
    #     if rand < 0.25 and x-1 >= 0:
    #             self.board[x-1,y] = self.CELL_TYPES_DICT["wall"]
    #     if rand < 0.125 and y+1 < self.height:
    #             self.board[x,y+1] = self.CELL_TYPES_DICT["wall"]
   
    def is_empty(self, coord):
        return self.CELL_TYPES_DICT["empty"] == self.board[coord]

    # returns the random location of an empty cell
    def find_empty(self):
        while True:
            coord = random.randint(0,self.boardsize[0]+1), random.randint(0,self.boardsize[1]+1)
            try:
                if not self.is_empty(coord):
                    raise IndexError("Co-ordinates are not empty")
            except:
                continue
            break
        return coord
        
    def collision_detection(self, coord, direction):
        """
        Method to check if new space is empty, if it is within the bounds and if there are any collision 
        with diagonal paths. Returns true if move is allowable

        Input:  path head, path direction
        Output: allowable move, new coordinates
        """
        new_coord = self._new_coord(coord, direction)
        x, y = coord
        x_new, y_new = new_coord
        # check new_coord is in bounds
        if x_new < 0 or x_new >= self.boardsize[0] or y_new < 0 or y_new >= self.boardsize[1]:
            return False, None
        # check coord is empty or not equal to the goal state
        elif not self.is_empty(new_coord) and self.board[new_coord] != self.board[coord]+1:
            return False, None
        # return True and new coords if direction is not diagonal
        elif direction in [self.ACTIONS_DICT["up"],self.ACTIONS_DICT["left"],self.ACTIONS_DICT["right"],self.ACTIONS_DICT["down"]]:
            return True, new_coord
        # check diagonal collision
        elif direction in [self.ACTIONS_DICT["upleft"], self.ACTIONS_DICT["upright"], self.ACTIONS_DICT["downleft"],self.ACTIONS_DICT["downright"]]:
            return self.is_empty((x_new, y)) or self.board[x_new, y] != self.board[x, y_new], new_coord
        else:
            return False, None

    def _new_coord(self, coord, direction):
        """
        Takes starting coordinates and an action and returns the new coordinates

        Inputs:  direction, starting coordinates
        Outputs: new coordinates
        """
        x, y = coord
        #if direction == self.ACTIONS_DICT["noop"]:
        #    return coord
        if direction == self.ACTIONS_DICT["upleft"]:
            return x-1, y+1
        elif direction == self.ACTIONS_DICT["up"]:
            return x, y+1
        elif direction == self.ACTIONS_DICT["upright"]:
            return x+1, y+1
        elif direction == self.ACTIONS_DICT["left"]:
            return x-1, y
        elif direction == self.ACTIONS_DICT["right"]:
            return x+1, y
        elif direction == self.ACTIONS_DICT["downleft"]:
            return x-1, y-1
        elif direction == self.ACTIONS_DICT["down"]:
            return x, y-1
        elif direction == self.ACTIONS_DICT["downright"]:
            return x+1, y-1
        else:
            raise IndexError("Index not a valid action")


class Game:

    def __init__(
        self, 
        boardsize=(10,20), 
        n_paths=5, 
        obstacles=15,
        partial_obs=None,
        starts=None, 
        goals=None,
        paddingsize=None
        ):

        self.obstacles = obstacles
        self.boardsize = boardsize
        self.paddingsize = paddingsize
        self.board = Board(obstacles=self.obstacles, boardsize=self.boardsize, 
                    paddingsize=self.paddingsize)
        self.paths = []
        self.n_paths = n_paths
        self.partial_obs = partial_obs

        # initialises the paths, and their head and goal locations
        for i in range(self.n_paths):
            if starts:
                start = starts[i]
                goal  = goals[i]
            else:
                start = self.board.find_empty()
                goal  = self.board.find_empty()
            path = Path(((i+1)*10)+10, start, goal)
            self.paths.append(path)
            self.board.change_cell(start, path.head)
            self.board.change_cell(goal, path.goal)
        self.starts = [x.start_coord for x in self.paths]
        self.goals = [x.goal_coord for x in self.paths]
    
    def step(self, directions):
        """
        Inputs:  list of global actions for each path
        Outputs: list new observations and 
                 list of path rewards 
                 boolean if the game is over
        """
        observation, reward = zip(*[self._path_step(self.paths[i], direction) for i, direction in enumerate(directions)])

        game_over = all([path.is_terminated() for path in self.paths])
        
        return observation, reward, game_over

    def _path_step(self, path, direction):
        """
        Contatins the reward function and termination conditions
        """
        allowable_move, new_coord = self.board.collision_detection(path.path_head(), direction)
        reward = 0

        # termination reward is a function of the relative board size.
        terminationreward = 25

        # if move is allowable
        if allowable_move:
            self.board.change_cell(path.path_head(), path.id)
            self.board.change_cell(new_coord, path.head)
            path.append_trail(new_coord)
            path.append_move(direction)
            # if the new coords are equal to the goal
            if new_coord == path.goal_coord:
                path.terminated = True
                reward += terminationreward
        # if move is not allowable terminate the path and give -25 reward
        else:
            path.terminated = True
            reward -= terminationreward

        # subtract -1 if direction is left, right, up, or down, else subtract root 2
        if direction in [self.board.ACTIONS_DICT["up"], self.board.ACTIONS_DICT["right"], self.board.ACTIONS_DICT["down"], self.board.ACTIONS_DICT["left"]]:
            reward -= 1
        else:
            reward -= (2**(1/2))
            
        # return the observation
        obs = self.get_observation(path)

        if self.partial_obs:
            x_y = path.x_y_distance()
            return (obs, x_y), reward
        else:
            return obs, reward

    def get_observation(self, path=None):
        if path:
            obs = self.board.get_partial_obs(path, self.partial_obs)
            xy = path.x_y_distance()
            return obs, xy
        else:
            return self.board.board

    def get_observation_dims(self):
        if self.partial_obs:
            return (self.partial_obs*2+1, self.partial_obs*2+1)
        else:
            return self.boardsize

    def get_actions(self):
        return len(self.board.ACTIONS_DICT)

    def get_state(self):
        return self.board.board

    def get_partial_obs(self):
        return self.partial_obs

    def render(self, save, path=None):
        plt.figure(figsize=(20,10))
        ax = plt.gca()
        ax.clear()
        clear_output(wait=True)
        env_plot = self.get_observation(path=path)
        colors = ['w','k','b','r','g','y','m','c','brown','navy','steelblue','grey']
        cmap = ListedColormap(colors[:self.n_paths+2])
        ax.matshow(env_plot,cmap=cmap)
        plt.savefig(f'trial{save}')
        plt.show()

    def reset(self):
        self.__init__(
            n_paths=self.n_paths, 
            boardsize=self.boardsize, 
            obstacles=self.obstacles, 
            starts=self.starts, 
            goals=self.goals,
            paddingsize=self.paddingsize,
            partial_obs=self.partial_obs
            )

class Path:

    def __init__(self, id: int, start_coord, goal_coord):
        if id % 10 != 0:
            raise ValueError("id must be divisible by 10")
        self.id = id
        self.head = self.id + 1
        self.goal = self.id + 2
        self.start_coord = start_coord
        self.goal_coord = goal_coord
        self.trail = [self.start_coord]
        self.moves = []
        self.terminated = False

    # returns the euclidean distance from the path head to its goal location
    def euclid_distance(self):
        x = self.path_head()[0] - self.goal_coord[0]
        y = self.path_head()[1] - self.goal_coord[1]
        return np.sqrt(x**2 + y**2)

    # returns the distance to goal in x and y directions
    def x_y_distance(self):
        x = self.path_head()[0] - self.goal_coord[0]
        y = self.path_head()[1] - self.goal_coord[1]
        return x,y

    def append_trail(self, new_coord):
        self.trail.append(new_coord)

    def append_move(self, direction):
        self.moves.append(direction)

    def path_head(self):
        return self.trail[-1]

    def is_terminated(self):
        return self.terminated
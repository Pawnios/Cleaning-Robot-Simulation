import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time
import gym


### Room generating functions ###

def initialize_room(width=10, max_island_size=5, min_n_islands=1, max_n_islands=5):
    """
    Initialises the basic room shape.
    Room is represented by a np.array(shape=(width, width)), initially just with 0 for wall and 1 for floor.
    Start by marking walls around the edges of the square, then creates 'islands' of inaccessible areas.
    No dirt is created yet. 
    """
    assert width > max_island_size
    
    # Initialize main room
    room = np.zeros([width, width], dtype=np.int8)
    
    # Create square to clean in center
    room[1:-1, 1:-1] = 1
    
    # Create random impassable 'islands'
    n_islands = np.random.randint(low=min_n_islands, high=max_n_islands)
    for island in range(n_islands):
        island_size = np.random.randint(low=1, high=max_island_size)
        island_x = np.random.randint(low=0-island_size + 1, high=width-1)
        island_y = np.random.randint(low=0-island_size + 1, high=width-1)
        
        for x_pos in range(island_x, island_x + island_size):
            for y_pos in range(island_y, island_y + island_size):
                if x_pos < 0:
                    x = 0
                elif x_pos >= width:
                    x = width - 1
                else:
                    x = x_pos
                if y_pos < 0:
                    y = 0
                elif y_pos >= width:
                    y = width - 1
                else:
                    y = y_pos
                
                room[x, y] = 0
    
    return room

def is_valid_room(room):
    """
    Takes in an initialised room of walls and floors, then returns True if every cell of the 
    room is accessible, False otherwise.
    """
    target_sum = np.sum(room)
    visited = np.zeros(room.shape)
    
    # If there are only walls, room is invalid
    if target_sum == 0:
        return False
    
    first_cell = np.argwhere(room==1)[0]
    
    def explore(room, current_cell, depth, max_depth=100):
        if depth > max_depth: return
        if visited[current_cell[0], current_cell[1]] == 1: return
        visited[current_cell[0], current_cell[1]] = 1
    
        neighbours = [[current_cell[0] - 1, current_cell[1]] if current_cell[0] > 0 else None, 
                      [current_cell[0] + 1, current_cell[1]] if current_cell[0] < room.shape[0] else None, 
                      [current_cell[0], current_cell[1] - 1] if current_cell[1] > 0 else None, 
                      [current_cell[0], current_cell[1] + 1] if current_cell[1] < room.shape[1] else None]
        neighbours = [neighbour for neighbour in neighbours if neighbour is not None]
        neighbours = [neighbour if room[neighbour[0], neighbour[1]] == 1 else None for neighbour in neighbours]
        neighbours = [neighbour for neighbour in neighbours if neighbour is not None]
        
        for neighbour in neighbours:
            explore(room, neighbour, depth + 1)
            
    explore(room, first_cell, depth=0)
    
    return np.sum(visited) == target_sum

def generate_room(width=10, max_island_size=5, min_n_islands=1, max_n_islands=5, seed=None):
    """
    Will generate a new room with given parameters.
    After 1 million attempts, program will throw an error, as it's likely the user has entered
    invalid parameters than cannot build a valid room.
    """
    if seed is not None:
        np.random.seed(seed)
    
    attempts = 1
    room = initialize_room(width, max_island_size, min_n_islands, max_n_islands)
    while not is_valid_room(room):
        assert attempts < 1e6, "1e6 generations attempted, issue with generation parameters."
        attempts += 1
        room = initialize_room(width, max_island_size, min_n_islands, max_n_islands)
        
    if seed is not None:
        reset_rng()
        
    return room


### Spawning functions ###

def spawn_robot(room, pos_x=None, pos_y=None, orientation=None, seed=None):
    """
    Spawns a robot into the room according to given coordinates, or 
    randomly if none are given.
    """
    # If robot spawn position is given
    if pos_x is not None and pos_y is not None:
        assert room[pos_x, pos_y] in [-1, 1], "Invalid spawn position."
        if orientation is None:
            orientation = np.random.randint(low=1, high=5)
        room[pos_x, pos_y] += orientation
        return room
    
    # Else, random or seeded spawn
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random spawn position and orientation
    room_size_x, room_size_y = room.shape[0], room.shape[1]
    pos_x, pos_y = np.random.randint(low=0, high=room_size_x), np.random.randint(low=0, high=room_size_y)
    while room[pos_x, pos_y] not in [-1, 1]:
        pos_x, pos_y = np.random.randint(low=0, high=room_size_x), np.random.randint(low=0, high=room_size_y)

    # An orientation of 1 is facing upward, then moving clockwise so 4 is nine o'clock
    orientation = np.random.randint(low=1, high=5) + 1  # +1 as it's spawning on floor, which has a value of 1
    room[pos_x, pos_y] = orientation
    
    if seed is not None:
        reset_rng()
    
    return room

def spawn_dirt(room, fraction=1, seed=None):
    """
    Creates dirt tiles in room. 
    The fraction parameter denotes the approximate fraction of tiles to be made dirty.
    """
    if seed is not None:
        np.random.seed(seed)
    
    existing_dirt = (room < 0).astype(np.int8)
    dirt = np.random.uniform(size=room.shape) + existing_dirt
    dirt = 2 * (dirt > fraction).astype(np.int8) - 1
    
    if seed is not None:
        reset_rng()
    
    robot_pos = np.argwhere(abs(room) > 1)
    if len(robot_pos) == 0:
        return dirt * room
    else:
        dirty_room = dirt * room
        dirty_room[robot_pos[0][0], robot_pos[0][1]] = abs(dirty_room[robot_pos[0][0], robot_pos[0][1]])
        return dirty_room
    
def spawn_n_dirt(room, n=1, seed=None):
    """
    Spawns n number of dirty tiles in room. Will always create this many more dirty tiles.
    """
    clean_tile_indices = np.argwhere(room == 1)
    
    n_clean_tiles = len(clean_tile_indices)
    
    # If there are not clean tiles to make dirty, return the room
    if n_clean_tiles == 0:
        return room
    
    if n_clean_tiles < n:
        n = n_clean_tiles
        
    if seed is not None:
        np.random.seed(seed)
    
    chosen_indices = clean_tile_indices[np.random.choice(len(clean_tile_indices), size=n, replace=False)]
    room[chosen_indices[:, 0], chosen_indices[:, 1]] = -1
    
    if seed is not None:
        reset_rng()
    
    return room

def clean_room(room):
    "Returns a cleaned version of the room."
    dirt = 2 * (room > 0).astype(np.int8) - 1
    return dirt * room


### Visualisation functions ###

def construct_image(room):
    is_robot_in_room = len(np.argwhere(abs(room) > 1)) > 0 
    
    # 0 for wall, 1 for clean floor, 2 for dirty floor, 3 for robot
    if is_robot_in_room:
        cmap = ListedColormap(['#2E282A', '#F3EFE0', '#A8664A','#80A1D4'])
    elif is_room_clean(room):
        cmap = ListedColormap(['#2E282A', '#F3EFE0'])
    else:
        cmap = ListedColormap(['#2E282A', '#F3EFE0', '#A8664A'])
    
    # Assume all is wall then build image
    image = np.zeros(shape=(room.shape))
    image[room > 0] = 1
    image[room < 0] = 2
    image[abs(room) > 1] = 3
    return image, cmap
    
def calculate_robot_arrow(room):
    # Draw arrow for robot orientation
    robot_position = np.argwhere(abs(room) > 1)[0]
    robot_orientation = abs(room[robot_position[0], robot_position[1]]) - 1

    # Draw a square to represent the robot
    orientation_map = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    dx, dy = orientation_map[robot_orientation]
    arrow = patches.FancyArrow(
        robot_position[1], 
        robot_position[0], 
        dx/4, 
        dy/4, 
        width=0.12, 
        head_width=0.4, 
        head_length=0.2, 
        color='#FFFFFF',
    )
    return arrow
    
def display_room(room):
    """
    Displays a room using matplotlib imshow with an arrow indicating orientation of robot.
    """
    
    image, cmap = construct_image(room)
    
    # Create image plot
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap, origin='lower')
    
    # If robot exists in the room
    if len(np.argwhere(abs(room) > 1)) > 0:
        # Get arrow which indicates the direction robot is facing
        arrow = calculate_robot_arrow(room)
        # Add the arrow to the plot
        ax.add_patch(arrow)
        
    plt.show()
    
    
### Logic functions ###

def is_room_clean(room):
    return -1 not in room


### RNG functions ###

def reset_rng():
    """
    Resets the NumPy random number generator with a new seed derived from the current time.
    This allows for unique seeding up to a 10th of a microsecond resolution within the 32-bit integer limit.
    """
    time_seed = np.random.seed(np.int64((time.time() * 1e7) % (2**32-1)))
    np.random.seed(time_seed)
    
    
### Robot movement functions ###
    
def get_robot_pos(room):
    robot_pos = np.argwhere(abs(room) > 1)
    if len(robot_pos) == 0:
        return None
    else:
        return robot_pos[0]
    
def robot_move_forward(room):
    """
    Move robot forward one cell.
    If against wall, robot remains in place. 
    Returns updated room and flag indicating if move was successful.
    """
    assert get_robot_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robot_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 1
    
    # Note that indices are rows, cols, so are [y, x] (not x, y)
    # Robot facing up
    if robot_orientation == 1: 
        move = np.array([1, 0])
    # Facing right
    elif robot_orientation == 2:
        move = np.array([0, 1])
    # Facing down
    elif robot_orientation == 3:
        move = np.array([-1, 0])
    # Facing left
    elif robot_orientation == 4:
        move = np.array([0, -1])
        
    new_pos = robot_pos + move
    
    if room[new_pos[0], new_pos[1]] == 0:
        return room, False
    
    # Place robot in new position
    room[new_pos[0], new_pos[1]] = room[robot_pos[0], robot_pos[1]]
    
    # Tile behind robot is clean
    room[robot_pos[0], robot_pos[1]] = 1
    
    return room, True

def robot_move_backward(room):
    "Same as move forward, just reversed."
    assert get_robot_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robot_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 1
    
    # Note that indices are rows, cols, so are [y, x] (not x, y)
    # Robot facing up
    if robot_orientation == 1: 
        move = np.array([-1, 0])
    # Facing right
    elif robot_orientation == 2:
        move = np.array([0, -1])
    # Facing down
    elif robot_orientation == 3:
        move = np.array([1, 0])
    # Facing left
    elif robot_orientation == 4:
        move = np.array([0, 1])
        
    new_pos = robot_pos + move
    
    if room[new_pos[0], new_pos[1]] == 0:
        return room, False
    
    # Place robot in new position
    room[new_pos[0], new_pos[1]] = room[robot_pos[0], robot_pos[1]]
    
    # Tile behind robot is clean
    room[robot_pos[0], robot_pos[1]] = 1
    
    return room, True

def robot_turn_right(room):
    "Robot rotates 90 degrees clockwise."
    assert get_robot_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robot_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 2  # -2 because -1 default, -1 again for mod 4
    
    new_orientation = ((robot_orientation + 1) % 4) + 2
    room[robot_pos[0], robot_pos[1]] = new_orientation
    return room, True

def robot_turn_left(room):
    "Robot rotates 90 degrees anti-clockwise."
    assert get_robot_pos(room) is not None, "No robot in room, move not possible."
    robot_pos = get_robot_pos(room)
    robot_orientation = room[robot_pos[0], robot_pos[1]] - 2  # -2 because -1 default, -1 again for mod 4
    
    new_orientation = ((robot_orientation - 1) % 4) + 2
    room[robot_pos[0], robot_pos[1]] = new_orientation
    return room, True

def robot_wait(room):
    "Robot doesn't perform any actions and just remains where it is."
    return room, True


# Gym environment class
class CleaningRobots:
    def __init__(self, config=None):
        # Environment configuration
        if config == None:
            self.config = self.default_config()
        else:
            self.config = config
        
        self.max_steps = self.config['max_steps']
        self.sparse_reward = self.config['sparse_reward']
        
        self.room = self.initialize_environment()
        
        # Action and observation spaces (6 in observation space is for number of channels we split space into)
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=-1, high=5, shape=(6, self.config['width'], self.config['width']), dtype=np.int8)

        # Create history for rendering video and calculating episode lengths
        self.history = []
        self.history.append(self.room.copy())

    
    def observe(self, decompose_channels=True):
        if decompose_channels:
            "Splits room up into several channels for an agent to learn from."
            accessible_channel = np.where((self.room != 0), 1, 0)                  # Channel containing acessible tiles
            dirt_channel = np.where((self.room == -1), 1, 0)                       # Channel containing location of dirt

            robot_orientation = np.max(self.room) - 2                              # 0 for up, 1 right, 2 down, 3 left
            robot_pos = np.argwhere(self.room == np.max(self.room))[0]             # Get robot position
            robot_pos_channels = np.zeros((4, *self.room.shape))                   # Create 4 channels for robot position
            robot_pos_channels[robot_orientation, robot_pos[0], robot_pos[1]] = 1  # Place position in corresponding channel and index

            return np.concatenate((np.stack((accessible_channel, dirt_channel)), robot_pos_channels), axis=0)
        else:
            return self.room.copy()
        
    def reset(self, seed=None):
        "Resets environment, generating a new room to clean according to config."
        self.config['seed'] = seed
        self.room = self.initialize_environment()
        self.history = []
        self.history.append(self.room.copy())
        return self.observe()
    
    def step(self, action):
        assert (not self.is_terminated()) and (not self.is_truncated()), "Environment is done."
        
        # Handle action first
        # Move forward
        if action == 0:
            self.room, action_success = robot_move_forward(self.room)
        
        # Move backward
        elif action == 1:
            self.room, action_success = robot_move_backward(self.room)
        
        # Turn right
        elif action == 2:
            self.room, action_success = robot_turn_right(self.room)
        
        # Turn left
        elif action == 3:
            self.room, action_success = robot_turn_left(self.room)
        
        # Wait
        elif action == 4:
            self.room, action_success = robot_wait(self.room)
        
        # Save new room to history
        self.history.append(self.room.copy()) 
        
        # Calculate reward
        reward = self.calculate_reward(action, action_success)
        
        # Placeholder
        info = {'step': len(self.history)}
        
        # Obs, reward, terminated, truncated, info
        return self.observe(), reward, self.is_terminated(), self.is_truncated() , info
            
    def render(self):
        "Plays video of current episode from start to now."
        fig, ax = plt.subplots()

        image_data, cmap = construct_image(self.history[0])
        img = ax.imshow(image_data, cmap=cmap, origin='lower')

        # Placeholder for the arrow patch
        arrow_patch = None

        def update(frame):
            nonlocal arrow_patch
            image_data, cmap = construct_image(self.history[frame])
            arrow_data = calculate_robot_arrow(self.history[frame])
            img.set_data(image_data)

            # Remove the previous arrow if it exists
            if arrow_patch is not None:
                arrow_patch.remove()

            # Add a new arrow patch
            arrow_patch = ax.add_patch(arrow_data)

            return [img, arrow_patch]

        ani = FuncAnimation(fig, update, frames=range(len(self.history)), interval=400, blit=True)
        plt.close()
        return ani
    
    def calculate_reward(self, action, action_success=True):
        """
        Sparse reward will give +1 reward if room is clean, also a small penalty for any move
        and a medium penalty for failing to perform the desired action, generally indicating collision.
        """
        assert self.config['sparse_reward'], 'Dense reward not implemented yet.'
        
        if self.config['sparse_reward']:
            # If an agent completes the largest possible room in this many steps, it's done a reasonable job.
            max_room_size = self.room.reshape(-1).shape[0]
            action_penalty_scale = 1 / (1 * (max_room_size * 2))
            
            # Penalty for any move, slightly larger penalty for moving backwards to discourage it
            if action in [0, 2, 3, 4]:
                action_penalty = -1 * action_penalty_scale
            elif action in [1]:
                action_penalty = -1.25 * action_penalty_scale
                
            # Fixed negative penalty for attempting an invalid action (colliding with an object)
            if action_success == False:
                action_penalty -= 0.1
                
            # If the room is clean
            if self.is_terminated():
                reward = 1.
                
            #  Worth experimenting with this. May lead to faster learning may not.
            #  elif self.is_truncated():
            #      reward = -1
                
            else:
                reward = 0.
                
            return reward + action_penalty
    
    def initialize_environment(self, attempts=0):
        """
        Builds an environment for the robot to clean.
        Initialises a room, then spawns the robot, then generates dirt according to the config.
        If initialisation fails (created done room for example) reinitialise, abandon after too many attempts.
        """
        self.room = generate_room(width=self.config['width'], 
                                  max_island_size=self.config['max_island_size'], 
                                  min_n_islands=self.config['min_n_islands'], 
                                  max_n_islands=self.config['max_n_islands'],
                                  seed=self.config['seed'])
        
        self.room = spawn_robot(self.room, 
                                seed=self.config['seed'])
        
        if self.config['n_dirt_generation']:
            self.room = spawn_n_dirt(self.room, 
                                     n=self.config['n_dirt_tiles'],
                                     seed=self.config['seed'])
        else:
            self.room = spawn_dirt(self.room, 
                               fraction=self.config['dirt_fraction'],
                               seed=self.config['seed'])
            
        # After 10,000 initialisation attempts, raise exception
        if attempts > 1e4:
            raise Exception("Max number of attempts to initialise environment exceeded.")
            
        if self.is_terminated():
            # If a room is created that is done, try again
            self.initialize_environment(attempts + 1)
            
        return self.room

    def default_config(self):
        """
        In case a config file isn't provided for the environment, these are the default settings.
        """
        config = {
            'width': 10,
            'max_island_size': 5,
            'min_n_islands': 1,
            'max_n_islands': 5,
            'dirt_fraction': 0.5,
            'n_dirt_generation': False,
            'n_dirt_tiles': 5,
            'seed': None,
            'max_steps': 1000,
            'sparse_reward': True,
        }
        return config
    
    def is_truncated(self):
        "Flag indicating environment has run for longer than max steps."
        return len(self.history) > self.max_steps
    
    def is_terminated(self):
        "Flag indicating room is clean."
        return -1 not in self.room

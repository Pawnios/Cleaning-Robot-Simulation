import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time
import gym



def initialize_room(width=10, max_island_size=5, min_n_islands=1, max_n_islands=5):

    assert width > max_island_size
    
    # Initialize main room
    room = np.zeros([width, width], dtype=np.int8)
    
    # Create square to clean in center
    room[1:-1, 1:-1] = 1
    n_islands = np.random.randint(low=min_n_islands, high=max_n_islands)
    for island in range(n_islands):
        island_size = np.random.randint(low=1, high=max_island_size)
        island_x = np.random.randint(low=0-island_size + 1, high=width-1)
        island_y = np.random.randint(low=0-island_size + 1, high=width-1)
        
        for x_pos in range(island_x, island_x + island_size):
            for y_pos in range(island_y, island_y + island_size):
                x = min(max(x_pos, 0), width - 1)
                y = min(max(y_pos, 0), width - 1)
                room[x, y] = 0
    
    return room

def is_valid_room(room):
    target_sum = np.sum(room == 1)
    visited = np.zeros(room.shape)
    if target_sum == 0:
        return False
    
    first_cell = np.argwhere(room==1)[0]
    
    def explore(room, current_cell, depth, max_depth=100):
        if depth > max_depth: return
        if visited[current_cell[0], current_cell[1]] == 1: return
        visited[current_cell[0], current_cell[1]] = 1
    
        neighbours = []
        if current_cell[0] > 0:
            neighbours.append([current_cell[0] - 1, current_cell[1]])
        if current_cell[0] < room.shape[0] - 1:
            neighbours.append([current_cell[0] + 1, current_cell[1]])
        if current_cell[1] > 0:
            neighbours.append([current_cell[0], current_cell[1] - 1])
        if current_cell[1] < room.shape[1] - 1:
            neighbours.append([current_cell[0], current_cell[1] + 1])
        neighbours = [n for n in neighbours if room[n[0], n[1]] == 1]
        
        for neighbour in neighbours:
            explore(room, neighbour, depth + 1)
            
    explore(room, first_cell, depth=0)
    
    return np.sum(visited) == target_sum


def add_dirt(room, n_dirt=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    floor_cells = np.argwhere(room == 1)
    dirt_indices = np.random.choice(len(floor_cells), min(n_dirt, len(floor_cells)), replace=False)
    for idx in dirt_indices:
        x, y = floor_cells[idx]
        room[x, y] = -1
    return room


def generate_room(width=10, max_island_size=5, min_n_islands=1, max_n_islands=5, seed=None, n_dirt=0):
    if seed is not None:
        np.random.seed(seed)
    attempts = 1
    room = initialize_room(width, max_island_size, min_n_islands, max_n_islands)
    while not is_valid_room(room):
        assert attempts < 1e6, "issue with generation parameters."
        attempts += 1
        room = initialize_room(width, max_island_size, min_n_islands, max_n_islands)
    if seed is not None:
        reset_rng()
    # Ensure add_dirt is defined elsewhere, or implement it as needed
    room = add_dirt(room, n_dirt=n_dirt, seed=seed)
    return room

def spawn_robot(room, pos_x=None, pos_y=None, orientation=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if pos_x is not None and pos_y is not None:
        assert room[pos_x, pos_y] in [-1, 1], "Invalid spawn position."
        if orientation is None:
            orientation = np.random.randint(1, 5)
        room[pos_x, pos_y] = 10 + orientation
        if seed is not None:
            reset_rng()
        return room

    room_size_x, room_size_y = room.shape[0], room.shape[1]
    pos_x, pos_y = np.random.randint(0, room_size_x), np.random.randint(0, room_size_y)
    while room[pos_x, pos_y] not in [-1, 1]:
        pos_x, pos_y = np.random.randint(0, room_size_x), np.random.randint(0, room_size_y)

    orientation = np.random.randint(1, 5)
    room[pos_x, pos_y] = 10 + orientation

    if seed is not None:
        reset_rng()

    return room

# def calculate_robot_arrow(room):
#     # Draw arrow for robot orientation
#     robot_position = np.argwhere(room >= 11)[0]
#     robot_orientation = room[robot_position[0], robot_position[1]] - 10  # 1-4
#     orientation_map = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
#     dx, dy = orientation_map.get(robot_orientation, (0, 1))
#     arrow = patches.FancyArrow(
#         robot_position[1], 
#         robot_position[0], 
#         dx/4, 
#         dy/4, 
#         width=0.12, 
#         head_width=0.4, 
#         head_length=0.2, 
#         color='#FFFFFF',
#     )
#     return arrow

def reset_rng():
    
    # Resets the NumPy random number generator with a new seed derived from the current time.
   
    np.random.seed(int((time.time() * 1e7) % (2**32-1)))
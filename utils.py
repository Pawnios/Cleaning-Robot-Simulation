import numpy as np

def room_to_channels(room):
    wall = (room == 0).astype(np.float32)
    dirt = (room == -1).astype(np.float32)
    robot = (room >= 10).astype(np.float32)
    return np.stack([wall, dirt, robot], axis=0)  # shape: (3, width, width)
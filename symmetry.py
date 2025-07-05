import numpy as np

def build_room_permutations(room):
    perms = [
        room.copy(),                              
        np.rot90(room, 1),                       
        np.rot90(room, 2),                        
        np.rot90(room, 3),                        
        np.fliplr(room),                         
        np.flipud(room),                         
        np.fliplr(np.rot90(room, 1)),           
        np.flipud(np.rot90(room, 1)),             
    ]
    return np.stack(perms)    
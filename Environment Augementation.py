def build_room_permutations(room):
    permutations =  np.stack((
        room.copy(),                  # Original
        room.copy().T[::-1],          # Rotated right 90 degrees
        room.copy()[::-1].T[::-1].T,  # Rotated 180 degrees
        room.copy()[::-1].T,          # Rotated right 270 degrees
        
        room.copy().T[::-1].T,        # Flipped horizontally
        room.copy().T,                # Flipped horizontally and rotated right 90 degrees
        room.copy()[::-1],            # Flipped horizontally and rotated 180 degrees
        room.copy()[::-1].T[::-1]     # Flipped horizontally and rotated right 270 degrees
    ))
    
    # Get robot positions 
    flat_indices = np.argmax(permutations.reshape(test_perms.shape[0], -1), axis=1)
    robot_positions = np.column_stack(np.unravel_index(flat_indices, permutations.shape[1:]))

    # Due to how we stored the robot positions, it's necessary we manually adjust the robots orientation
    rotations = np.array([0, 1, 2, 3, 0, 1, 2, 3])  # Indicating rotations of robot for each permutation
    permutations[np.arange(8), robot_positions[:, 0], robot_positions[:, 1]] = (
        (permutations[np.arange(8), robot_positions[:, 0], robot_positions[:, 1]] - 2 + rotations) % 4) + 2
        
    return permutations

def display_room_permutations(room):
    "Displays the original room, and the 7 identical permutations of that room."
    room_list = [
        room.copy(),
        room.copy().T[::-1],
        room.copy()[::-1].T[::-1].T,
        room.copy()[::-1].T,
        
        room.copy().T[::-1].T,
        room.copy().T,
        room.copy()[::-1],
        room.copy()[::-1].T[::-1]
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for idx, room in enumerate(room_list):
        
        robot_position = np.argwhere(abs(room) > 1)[0]
        current_orientation = room[robot_position[0], robot_position[1]]
        new_orientation = (current_orientation + idx - 2) % 4 + 2
        room[robot_position[0], robot_position[1]] = new_orientation
        
        image, cmap = construct_image(room)

        axes.flatten()[idx].imshow(image, cmap=cmap, origin='lower')

        # If robot exists in the room
        if len(np.argwhere(abs(room) > 1)) > 0:
            # Get arrow which indicates the direction robot is facing
            arrow = calculate_robot_arrow(room)
            # Add the arrow to the plot
            axes.flatten()[idx].add_patch(arrow)
        
    plt.tight_layout()
    plt.show()
    
display_room_permutations(env.room)

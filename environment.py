import numpy as np
import matplotlib.pyplot as plt
from symmetry import build_room_permutations

def display_room_permutations(room):
    """
    Displays the original room and its 7 symmetric permutations.
    """
    perms = build_room_permutations(room)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for idx, perm in enumerate(perms):
        axes.flatten()[idx].imshow(perm, cmap='gray', origin='lower')
        axes.flatten()[idx].set_title(f"Permutation {idx+1}")
        axes.flatten()[idx].axis('off')

    plt.tight_layout()
    plt.show()
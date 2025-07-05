# Cleaning-Robot-Simulation

# Objective

This project aims to develop a reinforcement learning environment simulating a robot tasked with cleaning dirty rooms, focusing on high efficiency.

# Features

Environment:

•Randomized room layout and dirt placement

•Gym API integration with step, observe, render, and reset methods

•Sparse reward function for effective learning

Agent:

•Vectorized random agent class for interaction

•Advanced image processing with policy and value neural networks inspired by convolutional neural network (CNN) architectures

# Planned Enhancements

•Curriculum Learning: Gradually increasing task complexity, starting from navigation to cleaning more dirt.

•Data Augmentation: Enhancing sample efficiency by leveraging rotational and reflectional symmetry.

# Channel Stacking

•Channel 1: Marks accessible tiles (1 for accessible, 0 for not)

•Channel 2: Identifies dirty tiles (1 for dirt, 0 for clean)

•Channels 3-6: Represent the robot's location and orientation (a single 1 in the corresponding position and channel)

# Augmentation

•Utilize room layout variations (rotations and flips) to generate 8 equivalent training samples.

•Store augmented states in GPU memory to perform rollouts, enhancing sample efficiency.

# Training Process

•Forward Pass: Collect logits and values from the augmented state tensor.

•Probability Distribution: Create distributions for each set of logits.

•Action Selection: Sample actions from a random permutation for each environment.

•Log Probability: Calculate log probabilities for each action in each permutation.

•Gradient Calculation: Compute gradients for each permutation using a single target value.

<img width="1440" alt="Screenshot 2024-07-16 at 4 19 28 PM" src="https://github.com/user-attachments/assets/c0ca1aaf-e6f1-4831-aa57-fcd20c279379">


# Modules Used

•gym: For creating the reinforcement learning environment

•numpy: For numerical operations

•torch: For building and training neural networks

•random: For random number generation

•matplotlib: For rendering the environment (optional)


# License

This project is licensed under the Apache 2.0 open source license.

# Acknowledgements

Gratitude to the reinforcement learning community for their invaluable resources.

from cleaning_robot_env import CleaningRobotEnv

env = CleaningRobotEnv(width=8)
obs = env.reset()
print("Observation shape:", obs.shape)  
env.render()

done = False
total_reward = 0
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    env.render()
    total_reward += reward

print("Episode finished. Total reward:", total_reward)